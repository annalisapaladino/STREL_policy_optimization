import torch

from drone.config import ScenarioCfg


DISCHARGE_RATE = 0.02
CHARGE_RATE = 0.10
ACTION_SCALE = 0.8
MOMENTUM =  0.6
CHARGE_DISTANCE = 0.8
CHARGE_SIGMOID_GAIN = 5.0
OCCUPANCY_DISTANCE = 0.7
OCCUPANCY_SIGMOID_GAIN = 10.0


def step_dynamics(curr_x, drone_actions, cfg: ScenarioCfg):
    d_pos_old = curr_x[cfg.drone_slice, 0:2]
    d_vel_old = curr_x[cfg.drone_slice, 2:4]
    d_vel_new = MOMENTUM * d_vel_old + (1.0 - MOMENTUM) * drone_actions
    d_pos_new = torch.clamp(d_pos_old + d_vel_new * ACTION_SCALE, 1.0, float(cfg.grid_side))

    b_pos = curr_x[cfg.base_slice, 0:2]
    dist_db = torch.norm(d_pos_new.unsqueeze(1) - b_pos.unsqueeze(0), dim=2)

    occupancy_per_drone = torch.sigmoid(OCCUPANCY_SIGMOID_GAIN * (OCCUPANCY_DISTANCE - dist_db))
    base_occupancy = torch.max(occupancy_per_drone, dim=0).values

    charging = torch.sigmoid(CHARGE_SIGMOID_GAIN * (CHARGE_DISTANCE - dist_db.min(dim=1).values))
    d_batt_old = curr_x[cfg.drone_slice, 4]
    d_batt_new = torch.clamp(d_batt_old - DISCHARGE_RATE + CHARGE_RATE * charging, 0.0, 1.0)

    # Build updated row blocks using torch.cat (out-of-place) to preserve the
    # computation graph.  In-place assignment to a no-grad clone would silently
    # sever the gradient connection from drone_actions through to world_pred.
    grid_block = curr_x[: cfg.n_grid]

    base_block = torch.cat(
        [curr_x[cfg.base_slice, 0:4], base_occupancy.unsqueeze(1), curr_x[cfg.base_slice, 5:]],
        dim=1,
    )

    drone_block = torch.cat(
        [d_pos_new, d_vel_new, d_batt_new.unsqueeze(1), curr_x[cfg.drone_slice, 5:]],
        dim=1,
    )

    return torch.cat([grid_block, base_block, drone_block], dim=0)


def step_dynamics_batched(curr_x, drone_actions, cfg: ScenarioCfg):
    """Batched version of step_dynamics.

    Args:
        curr_x:       (B, n_nodes, n_feat)
        drone_actions: (B, n_drones, 2)
    Returns:
        (B, n_nodes, n_feat)
    """
    d_pos_old = curr_x[:, cfg.drone_slice, 0:2]                           # (B, n_drones, 2)
    d_vel_old = curr_x[:, cfg.drone_slice, 2:4]                           # (B, n_drones, 2)
    d_vel_new = MOMENTUM * d_vel_old + (1.0 - MOMENTUM) * drone_actions
    d_pos_new = torch.clamp(d_pos_old + d_vel_new * ACTION_SCALE, 1.0, float(cfg.grid_side))

    b_pos = curr_x[:, cfg.base_slice, 0:2]                                # (B, n_bases, 2)
    dist_db = torch.norm(d_pos_new.unsqueeze(2) - b_pos.unsqueeze(1), dim=3)  # (B, n_drones, n_bases)

    occupancy_per_drone = torch.sigmoid(OCCUPANCY_SIGMOID_GAIN * (OCCUPANCY_DISTANCE - dist_db))
    base_occupancy = torch.max(occupancy_per_drone, dim=1).values          # (B, n_bases)

    charging = torch.sigmoid(CHARGE_SIGMOID_GAIN * (CHARGE_DISTANCE - dist_db.min(dim=2).values))
    d_batt_old = curr_x[:, cfg.drone_slice, 4]                            # (B, n_drones)
    d_batt_new = torch.clamp(d_batt_old - DISCHARGE_RATE + CHARGE_RATE * charging, 0.0, 1.0)

    grid_block = curr_x[:, : cfg.n_grid]                                  # (B, n_grid, n_feat)

    base_block = torch.cat(
        [curr_x[:, cfg.base_slice, 0:4], base_occupancy.unsqueeze(2), curr_x[:, cfg.base_slice, 5:]],
        dim=2,
    )                                                                      # (B, n_bases, n_feat)

    drone_block = torch.cat(
        [d_pos_new, d_vel_new, d_batt_new.unsqueeze(2), curr_x[:, cfg.drone_slice, 5:]],
        dim=2,
    )                                                                      # (B, n_drones, n_feat)

    return torch.cat([grid_block, base_block, drone_block], dim=1)        # (B, n_nodes, n_feat)


def rollout_batched(x0, actions_batch, cfg: ScenarioCfg):
    """Roll out B trajectories in parallel.

    Args:
        x0:            (n_nodes, n_feat)
        actions_batch: (B, T, n_drones, 2)
    Returns:
        (B, n_nodes, n_feat, T+1)  — ready for spec.quantitative
    """
    B, T, _, _ = actions_batch.shape
    s = x0.unsqueeze(0).expand(B, -1, -1)
    traj = [s]
    for t in range(T):
        s = step_dynamics_batched(s, actions_batch[:, t], cfg)
        traj.append(s)
    return torch.stack(traj).permute(1, 2, 3, 0)


def _init_base_positions(cfg: ScenarioCfg, device):
    if cfg.n_bases == 2:
        return torch.tensor([[3.0, 3.0], [7.0, 7.0]], device=device)
    xs = torch.linspace(2.0, cfg.grid_side - 2.0, cfg.n_bases, device=device)
    ys = torch.linspace(2.0, cfg.grid_side - 2.0, cfg.n_bases, device=device)
    return torch.stack([xs, ys], dim=1)
