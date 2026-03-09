import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from drone.config import ScenarioCfg
from drone.dynamics import _init_base_positions, step_dynamics_batched
from drone.specification import build_strel_specification
from policy.attention_policy import DroneAttentionPolicy
from policy.mlp_policy import DroneMLPPolicy, EgoDronePolicy
from stl.base import Node
from training.stats import append_epoch_stats


def _asymmetric_drift_loss(states_live: list, states_old: list, drone_slice: slice) -> torch.Tensor:
    """
    Cross-iteration asymmetric state drift penalty.

    Compares the live (current-policy) trajectory against the previous-iteration
    trajectory step by step.  A ReLU valve activates only when the new trajectory
    has *worse* battery (higher cost) than the old one; the penalty is then the
    mean squared drone-state distance.  This lets the optimiser make large jumps
    in parameter space when they improve performance, while pulling it back when
    they cause trajectory collapse.

    Args:
        states_live: list of T+1 tensors (B, n_nodes, n_feat) — live graph
        states_old:  list of T+1 tensors (B, n_nodes, n_feat) — detached
        drone_slice: slice selecting drone rows from the node dimension

    Returns:
        scalar drift loss, normalised by T
    """
    T = len(states_live) - 1
    drift = torch.zeros(1, device=states_live[1].device)
    for s_live, s_old in zip(states_live[1:], states_old[1:]):
        cost_live = -s_live[:, drone_slice, 4].mean()       # ↓battery → ↑cost; has grad
        cost_old  = -s_old[:, drone_slice, 4].mean()        # detached
        valve     = F.relu(cost_live - cost_old)            # 0 if improved, >0 if degraded
        dist_sq   = ((s_live[:, drone_slice, :5] - s_old[:, drone_slice, :5]) ** 2).mean()
        drift     = drift + valve * dist_sq
    return drift / T


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _load_checkpoint_payload(checkpoint_path: str | None, checkpoint: dict | None, runtime_device: torch.device):
    if checkpoint is not None:
        return checkpoint
    if checkpoint_path is None:
        return None
    return torch.load(checkpoint_path, map_location=runtime_device, weights_only=False)


def _sample_drone_init_positions(
    cfg,
    device,
    batch_size: int | None = None,
    init_pos_min: float = 1.0,
    init_pos_max: float | None = None,
) -> torch.Tensor:
    spawn_min = float(init_pos_min)
    spawn_max = float(cfg.grid_side) if init_pos_max is None else float(init_pos_max)

    if batch_size is None:
        pos = torch.rand((cfg.n_drones, 2), device=device) * (spawn_max - spawn_min) + spawn_min
    else:
        pos = torch.rand((batch_size, cfg.n_drones, 2), device=device) * (spawn_max - spawn_min) + spawn_min

    return pos


def _build_initial_state_batch(
    cfg,
    device,
    base_positions,
    grid_xy,
    batch_size: int,
    init_pos_min: float = 1.0,
    init_pos_max: float | None = None,
) -> torch.Tensor:
    x_state = torch.zeros((batch_size, cfg.n_nodes, cfg.n_feat), device=device)
    x_state[:, : cfg.n_grid, 0:2] = grid_xy.unsqueeze(0)
    x_state[:, : cfg.n_grid, 5] = 2.0

    x_state[:, cfg.base_slice, 0:2] = base_positions.unsqueeze(0)
    x_state[:, cfg.base_slice, 5] = 0.0

    x_state[:, cfg.drone_slice, 5] = 1.0
    x_state[:, cfg.drone_slice, 4] = 1.0

    x_state[:, cfg.drone_slice, 0:2] = _sample_drone_init_positions(
        cfg,
        device,
        batch_size=batch_size,
        init_pos_min=init_pos_min,
        init_pos_max=init_pos_max,
    )
    return x_state


def _trajectory_robustness_metrics(traj_batched: list[torch.Tensor], spec, cfg: ScenarioCfg):
    world = torch.stack(traj_batched, dim=0).permute(1, 2, 3, 0)
    rob_map = spec.quantitative(world, evaluate_at_all_times=True)
    min_rob_time = torch.min(rob_map[:, cfg.drone_slice, 0, :], dim=1).values
    mean_rob_per_env = torch.mean(min_rob_time, dim=1)
    return mean_rob_per_env, min_rob_time


def _build_policy(
    policy_type: str,
    cfg: ScenarioCfg,
    action_clip: float,
    policy_hidden: int,
    policy_res_blocks: int,
):
    policy_kind = str(policy_type).strip().lower()

    if policy_kind == "mlp":
        return DroneMLPPolicy(
            n_drones=cfg.n_drones,
            n_bases=cfg.n_bases,
            n_feat=cfg.n_feat,
            hidden=int(policy_hidden),
            out_channels=2,
            action_scale=action_clip,
        )

    if policy_kind == "ego":
        return EgoDronePolicy(
            hidden=int(policy_hidden),
            action_scale=action_clip,
        )

    if policy_kind == "attention":
        return DroneAttentionPolicy(
            n_feat=cfg.n_feat,
            action_scale=action_clip,
        )

    raise ValueError("policy_type must be one of: 'mlp', 'ego', 'attention'.")


def train_policy_gradient(
    grid_side: int = 10,
    n_bases: int = 2,
    n_drones: int = 1,
    n_feat: int = 6,
    T_total: int = 40,
    n_iterations: int = 1000,
    batch_size: int = 16,
    mini_batch_size: int | None = None,
    action_clip: float = 2.0,
    lr: float = 3e-4,
    lr_schedule: dict | None = None,
    grad_clip: float = 1.0,
    policy_type: str = "mlp",
    policy_hidden: int = 128,
    policy_res_blocks: int = 2,
    action_reg_weight: float = 1e-2,
    bptt_k: int | None = None,
    drift_weight: float = 0.0,
    smooth_beta_start: float | None = 1.0,
    smooth_beta_end: float = 50.0,
    smooth_beta_anneal_iters: int | None = None,
    exploration_std: float = 0.1,
    exploration_std_start: float | None = None,
    exploration_std_end: float | None = None,
    exploration_anneal_iters: int | None = None,
    init_pos_min: float = 1.0,
    init_pos_max: float | None = None,
    checkpoint_path: str | None = None,
    checkpoint: dict | None = None,
    save_checkpoint_path: str | None = None,
    save_best_checkpoint_path: str | None = None,
    device: str = "auto",
    show_progress: bool = True,
    return_history: bool = False,
):
    device_l = str(device).lower().strip()
    if device_l == "auto":
        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_l in ("cuda", "cpu"):
        if device_l == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available.")
        runtime_device = torch.device(device_l)
    else:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")

    cfg = ScenarioCfg(
        grid_side=int(grid_side),
        n_bases=int(n_bases),
        n_drones=int(n_drones),
        n_feat=int(n_feat),
    )
    spec = build_strel_specification(grid_side=cfg.grid_side)

    policy = _build_policy(
        policy_type=policy_type,
        cfg=cfg,
        action_clip=action_clip,
        policy_hidden=policy_hidden,
        policy_res_blocks=policy_res_blocks,
    ).to(runtime_device)

    n_policy_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {policy_type} | params: {n_policy_params:,}")

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Build sorted list of (milestone_iteration, lr) for manual step schedule
    _lr_milestones: list[tuple[int, float]] = []
    if lr_schedule:
        _lr_milestones = sorted((int(k), float(v)) for k, v in lr_schedule.items())
        print(f"LR schedule: {_lr_milestones}")

    checkpoint_payload = _load_checkpoint_payload(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        runtime_device=runtime_device,
    )

    start_iteration = 0
    stats_history = {
        "mean": [],
        "std": [],
        "median": [],
        "loss": [],
    }

    best_rob, worst_rob = -float("inf"), float("inf")
    best_traj, worst_traj = None, None
    best_rob_map, worst_rob_map = None, None
    best_policy_state_dict = None

    checkpoint_loaded = False
    checkpoint_type = None

    if checkpoint_payload is not None:
        if isinstance(checkpoint_payload, dict) and "policy_state_dict" in checkpoint_payload:
            policy.load_state_dict(checkpoint_payload["policy_state_dict"])

            if "optimizer_state_dict" in checkpoint_payload:
                optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
                _optimizer_to_device(optimizer, runtime_device)

            start_iteration = int(checkpoint_payload.get("iteration", 0))

            loaded_stats = checkpoint_payload.get("stats_history")
            if isinstance(loaded_stats, dict):
                for key in stats_history:
                    if key in loaded_stats:
                        stats_history[key] = list(loaded_stats[key])

            best_rob = float(checkpoint_payload.get("best_rob", best_rob))
            worst_rob = float(checkpoint_payload.get("worst_rob", worst_rob))

            best_traj = checkpoint_payload.get("best_traj", best_traj)
            worst_traj = checkpoint_payload.get("worst_traj", worst_traj)
            best_rob_map = checkpoint_payload.get("best_rob_map", best_rob_map)
            worst_rob_map = checkpoint_payload.get("worst_rob_map", worst_rob_map)

            checkpoint_loaded = True
            checkpoint_type = "full"
        else:
            policy.load_state_dict(checkpoint_payload)
            checkpoint_loaded = True
            checkpoint_type = "policy_state_dict"

    grid_xy = torch.stack(
        torch.meshgrid(
            torch.linspace(1, cfg.grid_side, cfg.grid_side, device=runtime_device),
            torch.linspace(1, cfg.grid_side, cfg.grid_side, device=runtime_device),
            indexing="ij",
        ),
        dim=2,
    ).reshape(-1, 2)
    base_positions = _init_base_positions(cfg, runtime_device)

    effective_mini_bs = batch_size if mini_batch_size is None else min(int(mini_batch_size), batch_size)

    use_smooth_beta = smooth_beta_start is not None
    if use_smooth_beta:
        smooth_beta_start_eff = float(smooth_beta_start)
        smooth_beta_end_eff = float(smooth_beta_end)
        smooth_beta_anneal_iters_eff = max(1, int(n_iterations if smooth_beta_anneal_iters is None else smooth_beta_anneal_iters))
        print(f"Smooth STL: beta {smooth_beta_start_eff} → {smooth_beta_end_eff} over {smooth_beta_anneal_iters_eff} iters")

    use_exploration_annealing = any(
        value is not None
        for value in (exploration_std_start, exploration_std_end, exploration_anneal_iters)
    )

    if use_exploration_annealing:
        exploration_std_start_eff = float(exploration_std if exploration_std_start is None else exploration_std_start)
        exploration_std_end_eff = float(exploration_std if exploration_std_end is None else exploration_std_end)
        exploration_anneal_iters_eff = max(1, int(n_iterations if exploration_anneal_iters is None else exploration_anneal_iters))
    else:
        exploration_std_start_eff = float(exploration_std)
        exploration_std_end_eff = float(exploration_std)
        exploration_anneal_iters_eff = 1

    iteration_iter = tqdm(
        range(start_iteration, start_iteration + n_iterations),
        desc="Policy Gradient Training",
        unit="iter",
        disable=not show_progress,
    )

    drift_prev_states: list | None = None  # detached states from previous iteration (on CPU)
    current_iteration = start_iteration
    for it in iteration_iter:
        if use_smooth_beta:
            anneal_progress = min(1.0, max(0.0, float(it) / float(smooth_beta_anneal_iters_eff)))
            current_beta = smooth_beta_start_eff + (smooth_beta_end_eff - smooth_beta_start_eff) * anneal_progress
        else:
            current_beta = None

        if use_exploration_annealing:
            anneal_progress = min(1.0, max(0.0, float(it) / float(exploration_anneal_iters_eff)))
            current_exploration_std = exploration_std_start_eff + (
                exploration_std_end_eff - exploration_std_start_eff
            ) * anneal_progress
        else:
            current_exploration_std = exploration_std

        # Manual step LR schedule: apply the lr for the highest milestone <= it
        if _lr_milestones:
            scheduled_lr = None
            for milestone, milestone_lr in _lr_milestones:
                if it >= milestone:
                    scheduled_lr = milestone_lr
            if scheduled_lr is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = scheduled_lr

        x0 = _build_initial_state_batch(
            cfg,
            runtime_device,
            base_positions,
            grid_xy,
            batch_size=batch_size,
            init_pos_min=init_pos_min,
            init_pos_max=init_pos_max,
        )
        current_iteration = it + 1

        # Gradient accumulation over mini-batches
        chunks = x0.split(effective_mini_bs, dim=0)  # list of (chunk_size, n_nodes, n_feat)
        n_samples = x0.shape[0]

        optimizer.zero_grad()
        total_loss = 0.0
        chunk_mean_robs = []
        chunk_min_rob_times = []
        chunk_states_det = []
        chunk_start = 0

        for chunk in chunks:
            chunk_size = chunk.shape[0]
            weight = chunk_size / n_samples

            # Differentiable rollout
            states = [chunk]
            actions_list = []
            s = chunk
            for t in range(T_total):
                a_mean = policy(s, None)
                if current_exploration_std > 0.0:
                    a = (a_mean + torch.randn_like(a_mean) * current_exploration_std).clamp(-action_clip, action_clip)
                else:
                    a = a_mean.clamp(-action_clip, action_clip)
                actions_list.append(a)
                s = step_dynamics_batched(s, a, cfg)
                if bptt_k is not None and (t + 1) % bptt_k == 0:
                    s = s.detach()
                states.append(s)

            # STL robustness (smooth semantics during training for dense gradients)
            world = torch.stack(states, dim=0).permute(1, 2, 3, 0)
            Node.set_smooth_beta(current_beta)
            rob_map = spec.quantitative(world, evaluate_at_all_times=True)
            Node.set_smooth_beta(None)
            min_rob_time = torch.min(rob_map[:, cfg.drone_slice, 0, :], dim=1).values  # (chunk_size, T+1)
            mean_rob_per_env = min_rob_time.mean(dim=1)                                  # (chunk_size,)

            # Action regularization
            action_reg = torch.stack([(a ** 2).mean() for a in actions_list]).mean()

            loss = -mean_rob_per_env.mean() + action_reg_weight * action_reg

            # Asymmetric state drift penalty: only penalises when trajectory degrades
            if drift_weight > 0.0 and drift_prev_states is not None:
                prev_chunk = [
                    drift_prev_states[t][chunk_start:chunk_start + chunk_size].to(runtime_device)
                    for t in range(T_total + 1)
                ]
                drift_loss = _asymmetric_drift_loss(states, prev_chunk, cfg.drone_slice)
                chunk_loss = loss * weight + drift_weight * drift_loss
            else:
                chunk_loss = loss * weight

            chunk_loss.backward()
            total_loss += float(loss.detach().item()) * weight

            chunk_mean_robs.append(mean_rob_per_env.detach())
            chunk_min_rob_times.append(min_rob_time.detach())
            chunk_states_det.append([st.detach().clone() for st in states])
            chunk_start += chunk_size

        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        # Aggregate metrics across chunks
        mean_rob_per_env = torch.cat(chunk_mean_robs)
        min_rob_time = torch.cat(chunk_min_rob_times, dim=0)
        states_full = [
            torch.cat([chunk_states_det[c][t] for c in range(len(chunks))], dim=0)
            for t in range(T_total + 1)
        ]

        # Store states for next-iteration drift penalty
        if drift_weight > 0.0:
            drift_prev_states = [st.cpu() for st in states_full]

        # Track best/worst trajectories
        epoch_rob_np = mean_rob_per_env.cpu().numpy()
        best_idx = int(torch.argmax(mean_rob_per_env).item())
        worst_idx = int(torch.argmin(mean_rob_per_env).item())

        best_candidate = float(mean_rob_per_env[best_idx].item())
        worst_candidate = float(mean_rob_per_env[worst_idx].item())

        if best_candidate > best_rob:
            best_rob = best_candidate
            best_traj = [states_full[t][best_idx].clone() for t in range(len(states_full))]
            best_rob_map = min_rob_time[best_idx].cpu().numpy()
            best_policy_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        if worst_candidate < worst_rob:
            worst_rob = worst_candidate
            worst_traj = [states_full[t][worst_idx].clone() for t in range(len(states_full))]
            worst_rob_map = min_rob_time[worst_idx].cpu().numpy()

        append_epoch_stats(stats_history, epoch_rob_np, total_loss)

        if show_progress:
            current_lr = optimizer.param_groups[0]["lr"]
            postfix = dict(
                mean_rob=f"{float(mean_rob_per_env.mean().item()):.4f}",
                loss=f"{total_loss:.4f}",
                lr=f"{current_lr:.2e}",
                exploration_std=f"{float(current_exploration_std):.4f}",
            )
            if current_beta is not None:
                postfix["beta"] = f"{current_beta:.1f}"
            iteration_iter.set_postfix(**postfix)

    if save_best_checkpoint_path is not None and best_policy_state_dict is not None:
        torch.save(best_policy_state_dict, save_best_checkpoint_path)

    if save_checkpoint_path is not None:
        checkpoint_out = {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": int(current_iteration),
            "stats_history": stats_history,
            "best_rob": best_rob,
            "worst_rob": worst_rob,
            "best_traj": best_traj,
            "worst_traj": worst_traj,
            "best_rob_map": best_rob_map,
            "worst_rob_map": worst_rob_map,
            "meta": {
                "algorithm": "policy_gradient_differentiable_dynamics",
                "grid_side": cfg.grid_side,
                "n_bases": cfg.n_bases,
                "n_drones": cfg.n_drones,
                "n_feat": cfg.n_feat,
                "T_total": T_total,
                "action_clip": action_clip,
                "policy_type": policy_type,
                "policy_hidden": policy_hidden,
                "policy_res_blocks": policy_res_blocks,
                "lr": lr,
                "grad_clip": grad_clip,
                "action_reg_weight": action_reg_weight,
                "bptt_k": bptt_k,
                "drift_weight": drift_weight,
                "smooth_beta_start": smooth_beta_start,
                "smooth_beta_end": smooth_beta_end,
                "smooth_beta_anneal_iters": smooth_beta_anneal_iters,
                "exploration_std": exploration_std,
                "exploration_std_start": exploration_std_start,
                "exploration_std_end": exploration_std_end,
                "exploration_anneal_iters": exploration_anneal_iters,
                "init_pos_min": init_pos_min,
                "init_pos_max": init_pos_max,
                "device": runtime_device.type,
            },
        }
        torch.save(checkpoint_out, save_checkpoint_path)

    if not return_history:
        return policy

    history = {
        "stats": stats_history,
        "best": {
            "robustness": best_rob,
            "trajectory": best_traj,
            "robustness_map": best_rob_map,
        },
        "worst": {
            "robustness": worst_rob,
            "trajectory": worst_traj,
            "robustness_map": worst_rob_map,
        },
        "config": {
            "algorithm": "policy_gradient_differentiable_dynamics",
            "grid_side": cfg.grid_side,
            "n_bases": cfg.n_bases,
            "n_drones": cfg.n_drones,
            "n_feat": cfg.n_feat,
            "T_total": T_total,
            "n_iterations": n_iterations,
            "start_iteration": start_iteration,
            "end_iteration": int(current_iteration),
            "batch_size": batch_size,
            "mini_batch_size": effective_mini_bs,
            "action_clip": action_clip,
            "policy_type": policy_type,
            "policy_hidden": policy_hidden,
            "policy_res_blocks": policy_res_blocks,
            "lr": lr,
            "grad_clip": grad_clip,
            "action_reg_weight": action_reg_weight,
            "bptt_k": bptt_k,
            "smooth_beta_start": smooth_beta_start,
            "smooth_beta_end": smooth_beta_end,
            "smooth_beta_anneal_iters": smooth_beta_anneal_iters,
            "exploration_std": exploration_std,
            "exploration_std_start": exploration_std_start,
            "exploration_std_end": exploration_std_end,
            "exploration_anneal_iters": exploration_anneal_iters,
            "init_pos_min": init_pos_min,
            "init_pos_max": init_pos_max,
            "drift_weight": drift_weight,
            "device": runtime_device.type,
            "n_policy_params": n_policy_params,
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_type": checkpoint_type,
            "checkpoint_path": checkpoint_path,
            "save_checkpoint_path": save_checkpoint_path,
        },
    }
    return policy, history


if __name__ == "__main__":
    train_policy_gradient()
