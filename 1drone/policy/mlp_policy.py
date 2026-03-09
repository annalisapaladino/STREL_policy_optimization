import torch
import torch.nn as nn


def flatten_obs_for_mlp(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        drone_mask = x[:, 5] == 1.0
        base_mask = x[:, 5] == 0.0

        drone_feats = x[drone_mask, :5].reshape(1, -1)
        base_feats = x[base_mask, :5].reshape(1, -1)
        return torch.cat([drone_feats, base_feats], dim=1)

    if x.dim() == 3:
        drone_mask = x[0, :, 5] == 1.0
        base_mask = x[0, :, 5] == 0.0

        drone_feats = x[:, drone_mask, :5].reshape(x.shape[0], -1)
        base_feats = x[:, base_mask, :5].reshape(x.shape[0], -1)
        return torch.cat([drone_feats, base_feats], dim=1)

    raise ValueError(f"Unsupported input shape for MLP policies: {tuple(x.shape)}")


class DroneMLPPolicy(nn.Module):
    """
    Simple MLP policy for drone control.

    Input: concatenation of [all drone features (excl. type), all base features (excl. type)]
           shape (n_drones * (n_feat-1) + n_bases * (n_feat-1),)
    Output: actions for all drones, shape (n_drones, 2)
    """

    def __init__(self, n_drones=2, n_bases=2, n_feat=6, hidden=128, out_channels=2, action_scale=2.0):
        super().__init__()
        in_dim = n_drones * (n_feat - 1) + n_bases * (n_feat - 1)
        self.n_drones = n_drones
        self.out_channels = out_channels
        self.action_scale = float(action_scale)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_drones * out_channels),
            nn.Tanh(),
        )

    def _flatten_obs(self, x: torch.Tensor) -> torch.Tensor:
        return flatten_obs_for_mlp(x)

    def forward(self, x, edge_index=None):
        # x: (n_nodes, n_feat) or (B, n_nodes, n_feat)
        # edge_index unused, kept for API compatibility
        obs = self._flatten_obs(x)
        batch_size = obs.shape[0]
        actions = self.net(obs).view(batch_size, self.n_drones, self.out_channels) * self.action_scale

        if x.dim() == 2:
            return actions.squeeze(0)
        return actions


class EgoDronePolicy(nn.Module):
    """
    Ego-centric policy for drone control.

    Instead of consuming absolute (drone_x, drone_y, base_x, base_y) and forcing
    the network to learn subtraction, we compute the relative displacement to the
    nearest base explicitly and feed only 5 meaningful features per drone:

        [rel_x, rel_y, vx, vy, battery]   (5 scalars)

    Benefits:
    - Translational invariance: the same weights work regardless of where on the
      grid the base is placed.
    - Drone-count invariance: weights are shared across drones via a per-drone
      forward pass, so the policy generalises to any number of drones.
    - ~93% fewer parameters than DroneMLPPolicy(hidden=128) for the same task.

    Parameters
    ----------
    hidden : int
        Hidden layer width (default 32 is sufficient given the compact input).
    action_scale : float
        Output is tanh × action_scale.
    """

    def __init__(self, hidden: int = 32, action_scale: float = 2.0):
        super().__init__()
        self.action_scale = float(action_scale)

        # Input: [rel_base_x, rel_base_y, vx, vy, battery]  →  5 features
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    def _build_ego_obs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build per-drone ego-centric observation.

        Args:
            x: (n_nodes, n_feat) or (B, n_nodes, n_feat)
        Returns:
            ego_obs: (B, n_drones, 5)
        """
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)  # (1, n_nodes, n_feat)

        # Node-type masks derived from the first batch element (same for all)
        drone_mask = x[0, :, 5] == 1.0   # (n_nodes,)
        base_mask  = x[0, :, 5] == 0.0   # (n_nodes,)

        drones = x[:, drone_mask, :]  # (B, n_drones, n_feat)
        bases  = x[:, base_mask,  :]  # (B, n_bases,  n_feat)

        d_pos  = drones[..., 0:2]    # (B, n_drones, 2)
        d_vel  = drones[..., 2:4]    # (B, n_drones, 2)
        d_batt = drones[..., 4:5]    # (B, n_drones, 1)
        b_pos  = bases[...,  0:2]    # (B, n_bases,  2)

        # Displacement vectors from every drone to every base: (B, n_drones, n_bases, 2)
        diffs = b_pos.unsqueeze(1) - d_pos.unsqueeze(2)

        # Select the nearest base for each drone
        nearest_idx = diffs.norm(dim=-1).argmin(dim=2)                    # (B, n_drones)
        B, D = nearest_idx.shape
        b_idx = torch.arange(B, device=x.device).view(-1, 1).expand(B, D)
        d_idx = torch.arange(D, device=x.device).view(1, -1).expand(B, D)
        rel_pos = diffs[b_idx, d_idx, nearest_idx]                        # (B, n_drones, 2)

        return torch.cat([rel_pos, d_vel, d_batt], dim=-1)                # (B, n_drones, 5)

    def forward(self, x, edge_index=None):
        # x: (n_nodes, n_feat) or (B, n_nodes, n_feat)
        is_batched = x.dim() == 3
        ego_obs = self._build_ego_obs(x)          # (B, n_drones, 5)
        actions = self.net(ego_obs) * self.action_scale   # (B, n_drones, 2)
        if not is_batched:
            return actions.squeeze(0)             # (n_drones, 2)
        return actions


