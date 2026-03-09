# ==============================================================================
# Copyright 2020-* Luca Bortolussi
# Copyright 2020-* Laura Nenzi
# AI-CPS Group @ University of Trieste
# ==============================================================================

import torch
import torch.nn.functional as F
from torch import Tensor


def _compute_euclidean_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between all agent pairs.
    x: [B, N, F, T], where [x,y,...].
    Returns: [B, T, N, N]
    """
    pos = x[:, :, 0:2, :].permute(0, 3, 1, 2)       # [B, T, N, 2]
    diffs = pos.unsqueeze(2) - pos.unsqueeze(3)       # [B, T, N, N, 2]
    return torch.norm(diffs, dim=-1)                  # [B, T, N, N]


def _change_coordinates(pos: Tensor, vel: Tensor, pos_agent: Tensor) -> Tensor:
    """Rotate relative coordinates into ego-centric frame."""
    x, y = pos
    vx, vy = vel
    xx, yy = pos_agent
    xx_pr = xx - x
    yy_pr = yy - y
    theta = torch.atan2(vy, vx + 1e-9)
    xx_sec = xx_pr * torch.cos(theta) + yy_pr * torch.sin(theta)
    yy_sec = -xx_pr * torch.sin(theta) + yy_pr * torch.cos(theta)
    return torch.stack((xx_sec, yy_sec), dim=0)  # [2]


def _compute_directional_distance_matrix(x: torch.Tensor, mode: str,
                                         side_thresh: float = 2.0) -> torch.Tensor:
    """
    Directional distance with side-threshold.
    x: [B,N,F,T], where [x,y,vx,vy,...].
    Returns: [B,T,N,N]
    """
    B, N, Fe, T = x.shape

    pos = x[:, :, 0:2, :].permute(0, 3, 1, 2)   # [B,T,N,2]
    vel = x[:, :, 2:4, :].permute(0, 3, 1, 2)   # [B,T,N,2]

    heading = F.normalize(vel, dim=-1, eps=1e-6)

    rel = pos.unsqueeze(2) - pos.unsqueeze(3)     # [B,T,N,N,2]

    # Projection along heading
    dot = (rel * heading.unsqueeze(2)).sum(-1)    # [B,T,N,N]

    # Lateral offset = norm of component orthogonal to heading
    heading_ortho = torch.stack([-heading[..., 1], heading[..., 0]], dim=-1)  # rotate 90°
    lateral = (rel * heading_ortho.unsqueeze(2)).sum(-1)  # [B,T,N,N]

    dist = torch.norm(rel, dim=-1)

    if mode == "Front":
        mask = (dot >= 0) & (lateral.abs() <= side_thresh)
    elif mode == "Back":
        mask = (dot <= 0) & (lateral.abs() <= side_thresh)
    elif mode == "Left":
        mask = (lateral >= 0) & (dot.abs() <= side_thresh)
    elif mode == "Right":
        mask = (lateral <= 0) & (dot.abs() <= side_thresh)
    else:
        raise ValueError(f"Unknown mode {mode}")

    return torch.where(mask, dist, torch.full_like(dist, float("inf")))


# Alias wrappers for compatibility
def _compute_front_distance_matrix(x): return _compute_directional_distance_matrix(x, "Front")
def _compute_back_distance_matrix(x):  return _compute_directional_distance_matrix(x, "Back")
def _compute_left_distance_matrix(x):  return _compute_directional_distance_matrix(x, "Left")
def _compute_right_distance_matrix(x): return _compute_directional_distance_matrix(x, "Right")


def compute_distance_matrix(x: Tensor, distance_function: str) -> Tensor:
    if distance_function == 'Euclid':
        return _compute_euclidean_distance_matrix(x)
    elif distance_function == 'Front':
        return _compute_front_distance_matrix(x)
    elif distance_function == 'Back':
        return _compute_back_distance_matrix(x)
    elif distance_function == 'Right':
        return _compute_right_distance_matrix(x)
    elif distance_function == 'Left':
        return _compute_left_distance_matrix(x)
    else:
        raise ValueError("Unknown distance function!!")


def _floyd_warshall_widest_path(C: Tensor) -> Tensor:
    """C: [B, T, N, N] edge capacities. Returns widest-path matrix."""
    W_cap = C.clone()
    n_nodes = W_cap.shape[-1]
    for k in range(n_nodes):
        w_ik = W_cap[:, :, :, k].clone()
        w_kj = W_cap[:, :, k, :].clone()
        cand = torch.minimum(w_ik.unsqueeze(-1), w_kj.unsqueeze(-2))
        W_cap = torch.maximum(W_cap, cand)
    return W_cap


def _floyd_warshall_shortest_path(W: Tensor, A: Tensor, POS_INF: float) -> Tensor:
    """W: [B, T, N, N] distances, A: adjacency. Returns all-pairs shortest paths."""
    d = torch.where(A.bool(), W, torch.tensor(POS_INF, device=W.device, dtype=W.dtype))
    n_nodes = d.shape[-1]
    mask = torch.eye(n_nodes, device=W.device, dtype=torch.bool)
    d = torch.where(mask.unsqueeze(0).unsqueeze(0), torch.tensor(0.0, device=W.device, dtype=W.dtype), d)

    for k in range(n_nodes):
        d_ik = d[:, :, :, k].clone()
        d_kj = d[:, :, k, :].clone()
        cand = d_ik.unsqueeze(-1) + d_kj.unsqueeze(-2)
        d = torch.minimum(d, cand)
    return d
