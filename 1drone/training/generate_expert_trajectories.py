import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone.config import ScenarioCfg
from drone.dynamics import _init_base_positions, step_dynamics, step_dynamics_batched
from drone.specification import build_strel_specification
from policy.attention_policy import DroneAttentionPolicy
from policy.mlp_policy import DroneMLPPolicy


@dataclass
class GenerationConfig:
    grid_side: int = 6
    n_bases: int = 1
    n_drones: int = 1
    n_feat: int = 6
    init_pos_min: float = 1.0
    init_pos_max: float | None = 6.0
    horizon: int = 100
    n_trajectories: int = 100
    max_attempts: int = 500
    robustness_threshold: float = 0.0
    n_mppi_iters: int = 100
    n_samples: int = 256
    sigma: float = 1.0
    sigma_decay: float = 0.95
    temperature: float = 0.01
    action_clip: float = 2.0
    action_reg_weight: float = 1e-2
    policy_path: str | None = "policy.pt"
    policy_type: str = "mlp"
    seed: int = 0
    device: str = "auto"
    save_every: int = 25
    keep_failed: bool = False


def _resolve_device(device: str) -> torch.device:
    d = device.lower().strip()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d in ("cpu", "cuda"):
        if d == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        return torch.device(d)
    raise ValueError("device must be one of: auto, cpu, cuda")


def _build_random_initial_state(
    cfg: ScenarioCfg,
    base_positions: torch.Tensor,
    device: torch.device,
    init_pos_min: float,
    init_pos_max: float | None,
) -> torch.Tensor:
    x = torch.zeros((cfg.n_nodes, cfg.n_feat), device=device)

    grid_xy = torch.stack(
        torch.meshgrid(
            torch.linspace(1, cfg.grid_side, cfg.grid_side, device=device),
            torch.linspace(1, cfg.grid_side, cfg.grid_side, device=device),
            indexing="ij",
        ),
        dim=2,
    ).reshape(-1, 2)

    x[: cfg.n_grid, 0:2] = grid_xy
    x[: cfg.n_grid, 5] = 2.0

    x[cfg.base_slice, 0:2] = base_positions
    x[cfg.base_slice, 5] = 0.0

    x[cfg.drone_slice, 5] = 1.0
    x[cfg.drone_slice, 4] = 1.0

    spawn_max = float(cfg.grid_side) if init_pos_max is None else float(init_pos_max)
    x[cfg.drone_slice, 0:2] = (
        torch.rand((cfg.n_drones, 2), device=device) * (spawn_max - float(init_pos_min)) + float(init_pos_min)
    )
    return x


def _rollout_open_loop(x0: torch.Tensor, actions: torch.Tensor, cfg: ScenarioCfg) -> list[torch.Tensor]:
    s = x0
    traj = [s]
    for t in range(actions.shape[0]):
        s = step_dynamics(s, actions[t], cfg)
        traj.append(s)
    return traj


def _trajectory_score_single(
    traj: list[torch.Tensor],
    spec,
    cfg: ScenarioCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    world = torch.stack(traj, dim=0).permute(1, 2, 0).unsqueeze(0)
    rob_map = spec.quantitative(world, evaluate_at_all_times=True)
    min_rob_time = torch.min(rob_map[:, cfg.drone_slice, 0, :], dim=1).values
    score = torch.mean(min_rob_time, dim=1).squeeze(0)
    return score, min_rob_time.squeeze(0)


def _policy_warm_start(
    policy: torch.nn.Module | None,
    x0: torch.Tensor,
    cfg: ScenarioCfg,
    horizon: int,
    action_clip: float,
    device: torch.device,
) -> torch.Tensor:
    if policy is None:
        return torch.zeros((horizon, cfg.n_drones, 2), device=device)

    with torch.no_grad():
        actions = torch.zeros((horizon, cfg.n_drones, 2), device=device)
        s = x0.unsqueeze(0)
        for t in range(horizon):
            a = policy(s, None).clamp(-action_clip, action_clip)
            actions[t] = a.squeeze(0)
            s = step_dynamics_batched(s, a, cfg)
    return actions


def _mppi_optimize(
    x0: torch.Tensor,
    spec,
    cfg: ScenarioCfg,
    horizon: int,
    n_iters: int,
    n_samples: int,
    sigma: float,
    temperature: float,
    action_clip: float,
    action_reg_weight: float,
    sigma_decay: float,
    U_init: torch.Tensor | None,
) -> tuple[torch.Tensor, float, list[float]]:
    device = x0.device
    U = U_init.clone().to(device) if U_init is not None else torch.zeros((horizon, cfg.n_drones, 2), device=device)

    best_score = -torch.inf
    best_actions = U.clone()
    score_trace: list[float] = []

    with torch.no_grad():
        for _ in range(n_iters):
            noise = torch.randn((n_samples, horizon, cfg.n_drones, 2), device=device) * sigma
            cand_actions = (U.unsqueeze(0) + noise).clamp(-action_clip, action_clip)

            s = x0.unsqueeze(0).expand(n_samples, -1, -1).clone()
            traj = [s]
            for t in range(horizon):
                s = step_dynamics_batched(s, cand_actions[:, t], cfg)
                traj.append(s)

            world = torch.stack(traj, dim=0).permute(1, 2, 3, 0)
            rob_map = spec.quantitative(world, evaluate_at_all_times=True)
            min_rob_time = torch.min(rob_map[:, cfg.drone_slice, 0, :], dim=1).values
            rob_scores = torch.mean(min_rob_time, dim=1)

            action_reg = cand_actions.pow(2).mean(dim=(1, 2, 3))
            scores = rob_scores - action_reg_weight * action_reg

            elite_idx = torch.argmax(scores)
            elite_score = scores[elite_idx]
            if elite_score > best_score:
                best_score = elite_score
                best_actions = cand_actions[elite_idx].detach().clone()

            baseline = torch.max(scores)
            weights = torch.softmax((scores - baseline) / max(1e-6, float(temperature)), dim=0)
            U = torch.sum(weights.view(-1, 1, 1, 1) * cand_actions, dim=0)

            score_trace.append(float(best_score.detach().item()))
            sigma = sigma * sigma_decay

    return best_actions, float(best_score.detach().item()), score_trace


def _build_policy(cfg: ScenarioCfg, policy_type: str, action_clip: float, device: torch.device) -> torch.nn.Module:
    if policy_type == "attention":
        return DroneAttentionPolicy(
            n_drones=cfg.n_drones,
            n_bases=cfg.n_bases,
            n_feat=cfg.n_feat,
            target_y_threshold=0.75 * cfg.grid_side,
            out_channels=2,
            action_scale=action_clip,
        ).to(device)

    if policy_type == "mlp":
        return DroneMLPPolicy(
            n_drones=cfg.n_drones,
            n_bases=cfg.n_bases,
            n_feat=cfg.n_feat,
            out_channels=2,
            action_scale=action_clip,
        ).to(device)

    raise ValueError("policy_type must be one of: mlp, attention")


def _load_optional_policy(
    cfg: ScenarioCfg,
    policy_type: str,
    policy_path: str | None,
    action_clip: float,
    device: torch.device,
) -> torch.nn.Module | None:
    if policy_path is None:
        return None

    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"policy_path does not exist: {path}")

    policy = _build_policy(cfg=cfg, policy_type=policy_type, action_clip=action_clip, device=device)
    state = torch.load(str(path), map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _save_checkpoint(
    output_path: Path,
    cfg: GenerationConfig,
    accepted: list[dict],
    rejected: list[dict],
    attempts: int,
    accepted_count: int,
    runtime_device: torch.device,
):
    payload = {
        "config": asdict(cfg),
        "device": runtime_device.type,
        "attempts": attempts,
        "accepted_count": accepted_count,
        "rejected_count": len(rejected),
        "accepted": accepted,
    }
    if cfg.keep_failed:
        payload["rejected"] = rejected

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output_path))


def generate_dataset(cfg: GenerationConfig, output_path: Path) -> None:
    torch.manual_seed(int(cfg.seed))
    runtime_device = _resolve_device(cfg.device)

    scenario = ScenarioCfg(
        grid_side=int(cfg.grid_side),
        n_bases=int(cfg.n_bases),
        n_drones=int(cfg.n_drones),
        n_feat=int(cfg.n_feat),
    )
    spec = build_strel_specification(grid_side=scenario.grid_side)
    base_positions = _init_base_positions(scenario, runtime_device)

    policy = _load_optional_policy(
        cfg=scenario,
        policy_type=cfg.policy_type,
        policy_path=cfg.policy_path,
        action_clip=float(cfg.action_clip),
        device=runtime_device,
    )

    accepted: list[dict] = []
    rejected: list[dict] = []

    progress = tqdm(total=int(cfg.n_trajectories), desc="Accepted experts", unit="traj")
    attempts = 0

    while len(accepted) < int(cfg.n_trajectories) and attempts < int(cfg.max_attempts):
        attempts += 1

        x0 = _build_random_initial_state(
            cfg=scenario,
            base_positions=base_positions,
            device=runtime_device,
            init_pos_min=float(cfg.init_pos_min),
            init_pos_max=cfg.init_pos_max,
        )

        U_policy = _policy_warm_start(
            policy=policy,
            x0=x0,
            cfg=scenario,
            horizon=int(cfg.horizon),
            action_clip=float(cfg.action_clip),
            device=runtime_device,
        )

        traj_policy = _rollout_open_loop(x0, U_policy, scenario)
        score_policy, _ = _trajectory_score_single(traj_policy, spec, scenario)

        best_actions, best_obj, _ = _mppi_optimize(
            x0=x0,
            spec=spec,
            cfg=scenario,
            horizon=int(cfg.horizon),
            n_iters=int(cfg.n_mppi_iters),
            n_samples=int(cfg.n_samples),
            sigma=float(cfg.sigma),
            temperature=float(cfg.temperature),
            action_clip=float(cfg.action_clip),
            action_reg_weight=float(cfg.action_reg_weight),
            sigma_decay=float(cfg.sigma_decay),
            U_init=U_policy,
        )

        traj_best = _rollout_open_loop(x0, best_actions, scenario)
        score_best, rob_trace_best = _trajectory_score_single(traj_best, spec, scenario)

        entry = {
            "x0": x0.detach().cpu(),
            "actions": best_actions.detach().cpu(),
            "trajectory": torch.stack(traj_best, dim=0).detach().cpu(),  # [T+1, n_nodes, n_feat]
            "robustness": float(score_best.detach().item()),
            "objective": float(best_obj),
            "policy_warm_start_robustness": float(score_policy.detach().item()),
            "robustness_trace": rob_trace_best.detach().cpu(),
        }

        if entry["robustness"] >= float(cfg.robustness_threshold):
            accepted.append(entry)
            progress.update(1)
        elif cfg.keep_failed:
            rejected.append(entry)

        if cfg.save_every > 0 and (len(accepted) > 0) and (len(accepted) % int(cfg.save_every) == 0):
            _save_checkpoint(
                output_path=output_path,
                cfg=cfg,
                accepted=accepted,
                rejected=rejected,
                attempts=attempts,
                accepted_count=len(accepted),
                runtime_device=runtime_device,
            )

        progress.set_postfix(
            attempts=attempts,
            accepted=len(accepted),
            accept_rate=f"{len(accepted) / max(1, attempts):.3f}",
        )

    progress.close()

    _save_checkpoint(
        output_path=output_path,
        cfg=cfg,
        accepted=accepted,
        rejected=rejected,
        attempts=attempts,
        accepted_count=len(accepted),
        runtime_device=runtime_device,
    )

    summary = {
        "output": str(output_path),
        "accepted": len(accepted),
        "requested": int(cfg.n_trajectories),
        "attempts": attempts,
        "accept_rate": (len(accepted) / max(1, attempts)),
        "threshold": float(cfg.robustness_threshold),
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate expert trajectories with MPPI + STL robustness objective.")

    p.add_argument("--output", type=str, required=True, help="Path to output .pt dataset file")

    p.add_argument("--grid-side", type=int, default=6)
    p.add_argument("--n-bases", type=int, default=1)
    p.add_argument("--n-drones", type=int, default=1)
    p.add_argument("--n-feat", type=int, default=6)
    p.add_argument("--init-pos-min", type=float, default=1.0)
    p.add_argument("--init-pos-max", type=float, default=6.0)

    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--n-trajectories", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=500)
    p.add_argument("--robustness-threshold", type=float, default=0.0)

    p.add_argument("--n-mppi-iters", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--sigma-decay", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--action-clip", type=float, default=2.0)
    p.add_argument("--action-reg-weight", type=float, default=1e-2)

    p.add_argument("--policy-path", type=str, default="policy.pt", help="Optional warm-start policy path; use '' to disable")
    p.add_argument("--policy-type", type=str, choices=["mlp", "attention"], default="mlp")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--keep-failed", action="store_true", help="Also store trajectories below threshold")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenerationConfig(
        grid_side=args.grid_side,
        n_bases=args.n_bases,
        n_drones=args.n_drones,
        n_feat=args.n_feat,
        init_pos_min=args.init_pos_min,
        init_pos_max=args.init_pos_max,
        horizon=args.horizon,
        n_trajectories=args.n_trajectories,
        max_attempts=args.max_attempts,
        robustness_threshold=args.robustness_threshold,
        n_mppi_iters=args.n_mppi_iters,
        n_samples=args.n_samples,
        sigma=args.sigma,
        sigma_decay=args.sigma_decay,
        temperature=args.temperature,
        action_clip=args.action_clip,
        action_reg_weight=args.action_reg_weight,
        policy_path=(None if args.policy_path.strip() == "" else args.policy_path),
        policy_type=args.policy_type,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        keep_failed=args.keep_failed,
    )

    if cfg.max_attempts < cfg.n_trajectories:
        raise ValueError("max_attempts must be >= n_trajectories")

    generate_dataset(cfg=cfg, output_path=Path(args.output))


if __name__ == "__main__":
    main()
