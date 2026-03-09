# ==============================================================================
# Copyright 2020-* Luca Bortolussi
# Copyright 2020-* Laura Nenzi
# AI-CPS Group @ University of Trieste
# ==============================================================================

"""Spatial STREL operators: Reach, Escape, Somewhere, Everywhere, Surround."""

import torch
import torch.nn.functional as F
from torch import Tensor

from stl.base import Node, realnum
from stl.logic import Not, And, Or
from stl.distance import (
    compute_distance_matrix,
    _floyd_warshall_widest_path,
    _floyd_warshall_shortest_path,
)


# ---------------------
#   VECTORIZED REACH
# ---------------------

class Reach(Node):
    """
    Vectorized Reach operator (multi-hop) with distance-bounded paths.

    Quantitative semantics (per batch/time/source node i):
        reach(i) = max_{j : d1 <= dist(i,j) <= d2} min( widest_path_capacity(i->j), s2(j) )

    - widest_path_capacity(i->j) is computed on the (max, min) semiring with node capacities = s1,
    - only nodes with left_label(s) are allowed as intermediates (via masking s1),
    - destinations must have right_label(s) (masked in s2).
    """

    def __init__(self,
                 left_child: Node,
                 right_child: Node,
                 d1: float,
                 d2: float,
                 left_label=None,       # int | list[int] | None
                 right_label=None,      # int | list[int] | None
                 is_unbounded: bool = False,
                 distance_domain_min: float = 0.,
                 distance_domain_max: float = float('inf'),
                 distance_function: str = 'Euclid'):
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.is_unbounded = is_unbounded
        self.distance_domain_min = float(distance_domain_min)
        self.distance_domain_max = float(distance_domain_max)

        self.distance_function = distance_function
        self.weight_matrix = None      # [B, T, N, N]
        self.adjacency_matrix = None   # [B, T, N, N] (0/1)
        self.num_nodes = None

        self.boolean_min_satisfaction = torch.tensor(0.0)
        self.quantitative_min_satisfaction = torch.tensor(float('-inf'))

        self.left_label = left_label
        self.right_label = right_label

        # numerically-stable sentinels
        self._NEG_INF = -1e9
        self._POS_INF = 1e9

    # -----------------------------
    # utilities
    # -----------------------------
    def _dist_fn(self, x: torch.Tensor) -> torch.Tensor:
        return compute_distance_matrix(x, self.distance_function)

    def _make_mask(self, lab: torch.Tensor, labels) -> torch.Tensor:
        """
        lab: [B,N,T] node types
        labels: int | list[int] | None
        """
        if labels is None:
            return torch.ones_like(lab, dtype=torch.bool, device=lab.device)
        if isinstance(labels, int):
            return (lab == labels)
        if isinstance(labels, (list, tuple)):
            mask = torch.zeros_like(lab, dtype=torch.bool)
            for l in labels:
                mask |= (lab == l)
            return mask
        raise ValueError("labels must be int, list[int], or None")

    def _initialize_matrices(self, x: torch.Tensor) -> None:
        device, dtype = x.device, x.dtype

        # x: [B, N, F, T]
        W = self._dist_fn(x).to(device=device, dtype=dtype)  # [B, T, N, N]
        A = (W > 0).to(x.dtype)                              # adjacency

        self.weight_matrix = W
        self.adjacency_matrix = A
        self.num_nodes = W.shape[-1]

        # channel -1 is type/label
        B, N, _, T = x.shape
        lab = x[:, :, -1, :]                                 # [B,N,T]

        self.left_mask = self._make_mask(lab, self.left_label)
        self.right_mask = self._make_mask(lab, self.right_label)

    # -----------------------------
    # Boolean via quantitative > 0
    # -----------------------------
    def _boolean(self, x: torch.Tensor) -> torch.Tensor:
        z = self._quantitative(x, normalize=False)
        return (z >= 0).to(torch.bool)

    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        self._initialize_matrices(x)

        device, dtype = x.device, x.dtype
        B, N, _, T = x.shape
        idx = torch.arange(N, device=device)

        NEG_INF = torch.tensor(self._NEG_INF, device=device, dtype=dtype)
        POS_INF = torch.tensor(self._POS_INF, device=device, dtype=dtype)

        # child signals
        s1 = self.left_child._quantitative(x, normalize).squeeze(2)
        s1 = torch.where(self.left_mask, s1, NEG_INF)  # forbid non-left-label intermediates

        s2 = self.right_child._quantitative(x, normalize).squeeze(2)
        s2 = torch.where(self.right_mask, s2, NEG_INF)  # forbid non-right-label destinations

        # edge capacities
        s1_btnt = s1.permute(0, 2, 1).contiguous()
        mask = torch.eye(N, device=device, dtype=torch.bool)
        C = torch.where(
            self.adjacency_matrix.bool(),
            s1_btnt.unsqueeze(-1).expand(B, T, N, N),
            NEG_INF
        )
        C = torch.where(mask.unsqueeze(0).unsqueeze(0),
            POS_INF,
                C)

        # widest path (max-min) Floyd-Warshall
        W_cap = _floyd_warshall_widest_path(C)

        # distances (min-plus) Floyd-Warshall
        D = _floyd_warshall_shortest_path(self.weight_matrix, self.adjacency_matrix, self._POS_INF)

        # distance window
        if self.is_unbounded:
            finite = torch.isfinite(D)
            finite_any = finite.any(dim=-1).any(dim=-1)  # [B,T]
            d2_eff = torch.where(
                finite_any,
                D.masked_fill(~finite, -POS_INF).amax(dim=-1).amax(dim=-1),
                torch.zeros(B, T, device=device, dtype=dtype)
            )
            lo = self.d1 - 1e-6
            elig = (D >= lo) & (D <= d2_eff.unsqueeze(-1).unsqueeze(-1) + 1e-6)
        else:
            elig = (D >= (self.d1 - 1e-6)) & (D <= (self.d2 + 1e-6))

        # combine
        s2_btnt = s2.permute(0, 2, 1)                        # [B,T,N]
        beta = self._smooth_beta
        if beta is None:
            pair_val = torch.minimum(W_cap, s2_btnt.unsqueeze(-2))
            pair_val = torch.where(elig, pair_val, NEG_INF)
            best_bt_n = pair_val.max(dim=-1).values           # [B,T,N]
        else:
            # Soft min(W_cap, s2): smooth approx of max-min semiring
            stk = torch.stack([W_cap, s2_btnt.unsqueeze(-2).expand_as(W_cap)], dim=-1)
            pair_val = -(1.0 / beta) * torch.logsumexp(-beta * stk, dim=-1)
            # Soft distance window: smooth penalty outside [d1, d2]
            soft_win = F.logsigmoid(beta * (self.d2 - D))
            if self.d1 > 1e-6:
                soft_win = soft_win + F.logsigmoid(beta * (D - self.d1))
            pair_val = pair_val + (1.0 / beta) * soft_win
            # Soft max over destinations
            best_bt_n = (1.0 / beta) * torch.logsumexp(beta * pair_val, dim=-1)
        return best_bt_n.permute(0, 2, 1).unsqueeze(2)        # [B,N,1,T]


# ---------------------
#     ESCAPE
# ---------------------

class Escape(Node):
    """
    Vectorized Escape operator (multi-hop) with distance-bounded paths.

    Quantitative semantics (per batch/time/source node i):
        escape(i) = max_{j : d1 <= dist(i,j) <= d2}  min( widest_path_capacity(i->j), s(j) )

    widest_path_capacity(i->j) is computed on the (max, min) semiring with node capacities = child,
    and only nodes with `labels` are allowed as intermediates (via masking the child signal).
    Destinations are also required to have `labels`.
    """

    def __init__(
        self,
        child: Node,
        d1: realnum,
        d2: realnum,
        labels: list = None,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid'
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = float(d1)
        self.d2 = float(d2)
        self.distance_domain_min = float(distance_domain_min)
        self.distance_domain_max = float(distance_domain_max)
        if labels is None:
            labels = []
        self.labels = labels
        self.distance_function = distance_function

        # Cached per-input
        self.weight_matrix = None    # [B, T, N, N]
        self.adjacency_matrix = None  # [B, T, N, N]
        self.num_nodes = None

        # numeric sentinels
        self._NEG_INF = -1e9
        self._POS_INF = 1e9

    def _dist_fn(self, x: Tensor) -> Tensor:
        return compute_distance_matrix(x, self.distance_function)

    def _initialize_matrices(self, x: Tensor) -> None:
        device, dtype = x.device, x.dtype

        W = self._dist_fn(x).to(device=device, dtype=dtype)  # [B, T, N, N]
        A = (W > 0).to(x.dtype)

        self.weight_matrix = W
        self.adjacency_matrix = A
        self.num_nodes = W.shape[-1]

        B, N, _, T = x.shape
        lab = x[:, :, -1, :]                                  # [B, N, T]
        if self.labels:
            m = torch.zeros_like(lab, dtype=torch.bool)
            for l in self.labels:
                m |= (lab == l)
            self.label_mask = m                                # [B, N, T]
        else:
            self.label_mask = torch.ones(B, N, T, dtype=torch.bool, device=device)

    def _boolean(self, x: Tensor) -> Tensor:
        return (self._quantitative(x, normalize=False) >= 0)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        """Returns [B, N, 1, T] robustness."""
        self._initialize_matrices(x)

        device, dtype = x.device, x.dtype
        B, N, _, T = x.shape

        NEG_INF = torch.tensor(self._NEG_INF, device=device, dtype=dtype)
        POS_INF = torch.tensor(self._POS_INF, device=device, dtype=dtype)

        # 1) Child signal as node capacity, masked by labels
        s = self.child._quantitative(x, normalize).squeeze(2)
        s = torch.where(self.label_mask, s, NEG_INF)

        # 2) Edge capacities C (use SOURCE node capacity): [B,T,N,N]
        s_btnt = s.permute(0, 2, 1).contiguous()              # [B, T, N]
        mask = torch.eye(N, device=device, dtype=torch.bool)
        C = torch.where(
            self.adjacency_matrix.bool(),
            s_btnt.unsqueeze(-1).expand(B, T, N, N),
            NEG_INF
        )
        C = torch.where(mask.unsqueeze(0).unsqueeze(0),
                        POS_INF,
                        C)
        C = C.clone()

        # 3) Widest-path (max-min) via Floyd-Warshall on C
        W_cap = _floyd_warshall_widest_path(C)

        # 4) All-pairs shortest path distances (min-plus) for the window
        D = _floyd_warshall_shortest_path(self.weight_matrix, self.adjacency_matrix, self._POS_INF)

        # 5) Distance eligibility mask within [d1, d2]
        elig = (D >= (self.d1 - 1e-6)) & (D <= (self.d2 + 1e-6))  # [B,T,N,N]

        # 6) Combine widest-path capacity with destination capacity s(j)
        s_dest_btnt = s.permute(0, 2, 1)                      # [B,T,N]
        pair_val = torch.minimum(W_cap, s_dest_btnt.unsqueeze(-2))  # [B,T,N,N]
        pair_val = torch.where(elig, pair_val, NEG_INF)

        # 7) For each source i: max over destinations j
        best_bt_n = pair_val.max(dim=-1).values                # [B,T,N]

        # 8) Return [B,N,1,T]
        return best_bt_n.permute(0, 2, 1).unsqueeze(2)


# ---------------------
#     SOMEWHERE
# ---------------------

class Somewhere(Node):
    """
    Somewhere operator for STREL. Models existence of a satisfying location within a distance interval.
    Equivalent to Reach(True, phi).
    """
    def __init__(
        self,
        child: Node,
        d2: realnum,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid',
        labels: list = None
    ) -> None:
        super().__init__()
        from stl.base import Atom
        self.child = child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.distance_function = distance_function
        if labels is None:
            labels = []
        self.labels = labels

        # True node (always satisfied)
        self.true_node = Atom(0, float('inf'), lte=True)

        # Reach(True, phi)
        self.reach_op = Reach(
            left_child=self.true_node,
            right_child=child,
            d1=self.d1,
            d2=d2,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=self.distance_function,
            left_label=[],
            right_label=self.labels
        )

    def __str__(self) -> str:
        return f"somewhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.reach_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.reach_op._quantitative(x, normalize)


# ---------------------
#     EVERYWHERE
# ---------------------

class Everywhere(Node):
    """
    Everywhere operator for STREL.
    Equivalent to NOT Somewhere(NOT phi).
    """
    def __init__(
        self,
        child: Node,
        d2: realnum,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid',
        labels: list = None
    ) -> None:
        super().__init__()
        self.child = child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.distance_function = distance_function
        if labels is None:
            labels = []
        self.labels = labels

        # Everywhere phi := NOT Somewhere(NOT phi)
        self.somewhere_op = Somewhere(
            child=Not(self.child),
            d2=d2,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=self.distance_function,
            labels=self.labels
        )
        self.everywhere_op = Not(self.somewhere_op)

    def __str__(self) -> str:
        return f"everywhere_[{self.d1},{self.d2}] ( {self.child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.everywhere_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.everywhere_op._quantitative(x, normalize)


# ---------------------
#     SURROUND
# ---------------------

class Surround(Node):
    """
    Surround operator for STREL.
    phi1 SURROUNDED_BY phi2 within distance <= d2.
    """
    def __init__(
        self,
        left_child: Node,
        right_child: Node,
        d2: realnum,
        distance_domain_min: realnum = 0.,
        distance_domain_max: realnum = float('inf'),
        distance_function: str = 'Euclid',
        left_labels: list = None,
        right_labels: list = None,
        all_labels: list = None
    ) -> None:
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.d1 = 0
        self.d2 = d2
        self.distance_domain_min = distance_domain_min
        self.distance_domain_max = distance_domain_max
        self.distance_function = distance_function

        if left_labels is None:
            left_labels = []
        if right_labels is None:
            right_labels = []
        if all_labels is None:
            all_labels = []

        # Copy to avoid in-place mutation
        all_labels = list(all_labels)
        for l in left_labels:
            if l in all_labels:
                all_labels.remove(l)
        for r in right_labels:
            if r in all_labels:
                all_labels.remove(r)

        self.complementary_labels = all_labels

        # Reach( phi1 , NOT(phi1 OR phi2) )
        self.reach_op = Reach(
            left_child=self.left_child,
            right_child=Not(Or(self.left_child, self.right_child)),
            d1=self.d1, d2=d2,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=distance_function,
            left_label=left_labels,
            right_label=self.complementary_labels
        )
        self.neg_reach = Not(self.reach_op)

        # Escape(phi1)
        self.escape_op = Escape(
            child=self.left_child,
            d1=d2, d2=distance_domain_max,
            distance_domain_min=distance_domain_min,
            distance_domain_max=distance_domain_max,
            distance_function=distance_function,
            labels=left_labels
        )
        self.neg_escape = Not(self.escape_op)

        self.right_labels = right_labels
        self.left_labels = left_labels

        # (phi1 AND NOT Reach) AND NOT Escape
        self.conj1 = And(self.left_child, self.neg_reach)
        self.surround_op = And(self.conj1, self.neg_escape)

    def __str__(self):
        return f"surround_[{self.d1},{self.d2}] ( {self.left_child} , {self.right_child} )"

    def _boolean(self, x: Tensor) -> Tensor:
        return self.surround_op._boolean(x)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        return self.surround_op._quantitative(x, normalize)
