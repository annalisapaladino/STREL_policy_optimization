# ==============================================================================
# Copyright 2020-* Luca Bortolussi
# Copyright 2020-* Laura Nenzi
# AI-CPS Group @ University of Trieste
# ==============================================================================

"""Temporal STL operators: Globally, Eventually, Until, Since."""

import torch
import torch.nn.functional as F
from torch import Tensor

from stl.base import Node
from stl.logic import And, Or, Not


def _soft_cumlse_from_right(z: Tensor, sign: float, beta: float) -> Tensor:
    """Cumulative log-sum-exp from the right along the last dimension.

    sign=+1 → soft cummax from right  (Eventually):  result[...,t] = (1/β) * logsumexp( β * z[..., t:])
    sign=-1 → soft cummin from right  (Globally):    result[...,t] = -(1/β) * logsumexp(-β * z[..., t:])

    As β → ∞ converges to cummax / cummin respectively.
    """
    scaled_flip = sign * beta * torch.flip(z, [-1])
    T = z.shape[-1]
    running = scaled_flip[..., 0]
    slices = [running.unsqueeze(-1)]
    for t in range(1, T):
        running = torch.logaddexp(running, scaled_flip[..., t])
        slices.append(running.unsqueeze(-1))
    cum_lse = torch.cat(slices, dim=-1)
    return (sign / beta) * torch.flip(cum_lse, [-1])


def _soft_pool(z: Tensor, k: int, beta: float) -> Tensor:
    """Soft max pool: (1/β) * logsumexp(β * z_window) over windows of size k along last dim.

    Replaces F.max_pool1d for bounded Eventually.
    For bounded Globally use -_soft_pool(-z, k, β).
    """
    if z.dim() == 4:
        B, N, C, T = z.shape
        z_r = z.reshape(B * N, C, T)
    elif z.dim() == 3:
        B, N, T = z.shape
        z_r = z.reshape(B * N, 1, T)
    else:
        raise ValueError(f"Unsupported shape {z.shape} for _soft_pool")
    windows = z_r.unfold(-1, k, 1)          # [..., C, T_new, k]
    soft = (1.0 / beta) * (beta * windows).logsumexp(dim=-1)  # [..., C, T_new]
    T_new = soft.shape[-1]
    if z.dim() == 4:
        return soft.reshape(B, N, C, T_new)
    return soft[:, 0, :].reshape(B, N, T_new)


def eventually(x: Tensor, time_span: int) -> Tensor:
    """
    STL operator 'eventually' applied along the last dimension (time).
    x: [B, N, 1, T] or [B, N, T]
    returns: same shape as input
    """
    if x.dim() == 4:   # [B,N,1,T]
        B, N, C, T = x.shape
        x_reshaped = x.reshape(B * N, C, T)
        y = F.max_pool1d(x_reshaped, kernel_size=time_span, stride=1)
        T_new = y.shape[-1]
        return y.reshape(B, N, C, T_new)
    elif x.dim() == 3:  # [B,N,T]
        B, N, T = x.shape
        x_reshaped = x.reshape(B * N, 1, T)
        y = F.max_pool1d(x_reshaped, kernel_size=time_span, stride=1)
        T_new = y.shape[-1]
        return y.reshape(B, N, T_new)
    else:
        raise ValueError(f"Unsupported input shape {x.shape} for eventually()")


# ---------------------
#     GLOBALLY
# ---------------------

class Globally(Node):
    """Globally node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "always" + s0 + " ( " + self.child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            return self.child.time_depth() + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummin(torch.flip(z1, [3]), dim=3)
                z: Tensor = torch.flip(z, [3])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.min(z1, 3, keepdim=True)
        else:
            z: Tensor = torch.ge(1.0 - eventually((~z1).double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, :, self.left_time_bound:], normalize)
        beta = self._smooth_beta
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                if beta is None:
                    z: Tensor
                    _: Tensor
                    z, _ = torch.cummin(torch.flip(z1, [3]), dim=3)
                    z: Tensor = torch.flip(z, [3])
                else:
                    z: Tensor = _soft_cumlse_from_right(z1, sign=-1.0, beta=beta)
            else:
                if beta is None:
                    z: Tensor
                    _: Tensor
                    z, _ = torch.min(z1, 3, keepdim=True)
                else:
                    z: Tensor = -(1.0 / beta) * torch.logsumexp(-beta * z1, dim=3, keepdim=True)
        else:
            k = self.right_time_bound - self.left_time_bound
            if beta is None:
                z: Tensor = -eventually(-z1, k)
            else:
                z: Tensor = -_soft_pool(-z1, k, beta)
        return z


# ---------------------
#     EVENTUALLY
# ---------------------

class Eventually(Node):
    """Eventually node."""

    def __init__(
            self,
            child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "eventually" + s0 + " ( " + self.child.__str__() + " )"
        return s

    # TODO: coherence between computation of time depth and time span given when computing eventually 1d
    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            return self.child.time_depth() + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, :, self.left_time_bound:])
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                z: Tensor
                _: Tensor
                z, _ = torch.cummax(torch.flip(z1, [3]), dim=3)
                z: Tensor = torch.flip(z, [3])
            else:
                z: Tensor
                _: Tensor
                z, _ = torch.max(z1, 3, keepdim=True)
        else:
            z: Tensor = torch.ge(eventually(z1.double(), self.right_time_bound - self.left_time_bound), 0.5)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, :, self.left_time_bound:], normalize)
        beta = self._smooth_beta
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                if beta is None:
                    z: Tensor
                    _: Tensor
                    z, _ = torch.cummax(torch.flip(z1, [3]), dim=3)
                    z: Tensor = torch.flip(z, [3])
                else:
                    z: Tensor = _soft_cumlse_from_right(z1, sign=1.0, beta=beta)
            else:
                if beta is None:
                    z: Tensor
                    _: Tensor
                    z, _ = torch.max(z1, 3, keepdim=True)
                else:
                    z: Tensor = (1.0 / beta) * torch.logsumexp(beta * z1, dim=3, keepdim=True)
        else:
            k = self.right_time_bound - self.left_time_bound
            if beta is None:
                z: Tensor = eventually(z1, k)
            else:
                z: Tensor = _soft_pool(z1, k, beta)
        return z


# ---------------------
#     UNTIL
# ---------------------

class Until(Node):
    # TODO: maybe define timed and untimed until, and use this class to wrap them
    # TODO: maybe faster implementation (of untimed until especially)
    """Until node."""

    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = "( " + self.left_child.__str__() + " until" + s0 + " " + self.right_child.__str__() + " )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _make_timed_until_node(self) -> Node:
        if self.right_unbound:
            return And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                       And(Eventually(self.right_child, right_unbound=True,
                                      left_time_bound=self.left_time_bound),
                           Eventually(Until(self.left_child, self.right_child, unbound=True),
                                      left_time_bound=self.left_time_bound, right_unbound=True)))
        return And(Globally(self.left_child, left_time_bound=0, right_time_bound=self.left_time_bound),
                   And(Eventually(self.right_child, left_time_bound=self.left_time_bound,
                                  right_time_bound=self.right_time_bound - 1),
                       Eventually(Until(self.left_child, self.right_child, unbound=True),
                                  left_time_bound=self.left_time_bound, right_unbound=True)))

    def _boolean(self, x: Tensor) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._boolean(x)
            z2: Tensor = self.right_child._boolean(x)
            size: int = min(z1.size(3), z2.size(3))
            z1: Tensor = z1[:, :, :, :size]
            z2: Tensor = z2[:, :, :, :size]
            z1_rep = torch.repeat_interleave(z1.unsqueeze(3), z1.unsqueeze(3).shape[-1], 3)
            z1_tril = torch.tril(z1_rep.transpose(3, 4), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=4)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(3), z2.unsqueeze(3).shape[-1], 3)
            z2_tril = torch.tril(z2_rep.transpose(3, 4), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
        else:
            timed_until: Node = self._make_timed_until_node()
            z: Tensor = timed_until._boolean(x)
        return z

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        if self.unbound:
            z1: Tensor = self.left_child._quantitative(x, normalize)
            z2: Tensor = self.right_child._quantitative(x, normalize)
            size: int = min(z1.size(3), z2.size(3))
            z1: Tensor = z1[:, :, :, :size]
            z2: Tensor = z2[:, :, :, :size]

            z1_rep = torch.repeat_interleave(z1.unsqueeze(3), z1.unsqueeze(3).shape[-1], 3)
            z1_tril = torch.tril(z1_rep.transpose(3, 4), diagonal=-1)
            z1_triu = torch.triu(z1_rep)
            z1_def = torch.cummin(z1_tril + z1_triu, dim=4)[0]

            z2_rep = torch.repeat_interleave(z2.unsqueeze(3), z2.unsqueeze(3).shape[-1], 3)
            z2_tril = torch.tril(z2_rep.transpose(3, 4), diagonal=-1)
            z2_triu = torch.triu(z2_rep)
            z2_def = z2_tril + z2_triu
            z: Tensor = torch.max(torch.min(torch.cat([z1_def.unsqueeze(-1), z2_def.unsqueeze(-1)], dim=-1), dim=-1)[0],
                                  dim=-1)[0]
        else:
            timed_until: Node = self._make_timed_until_node()
            z: Tensor = timed_until._quantitative(x, normalize=normalize)
        return z


# ---------------------
#     SINCE
# ---------------------

class Since(Node):
    """Since node (past-time Until: phi_1 S[a,b] phi_2).

    Simulated by flipping the time axis and reusing Until semantics.
    """

    def __init__(
            self,
            left_child: Node,
            right_child: Node,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
    ) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1

        if (not self.unbound) and (not self.right_unbound) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        s: str = f"( {self.left_child} since{s0} {self.right_child} )"
        return s

    def time_depth(self) -> int:
        sum_children_depth: int = self.left_child.time_depth() + self.right_child.time_depth()
        if self.unbound:
            return sum_children_depth
        elif self.right_unbound:
            return sum_children_depth + self.left_time_bound
        else:
            return sum_children_depth + self.right_time_bound - 1

    def _boolean(self, x: Tensor) -> Tensor:
        x_flipped = torch.flip(x, [3])
        until_node = Until(
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )
        z_flipped = until_node._boolean(x_flipped)
        return torch.flip(z_flipped, [3])

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        x_flipped = torch.flip(x, [3])
        until_node = Until(
            self.left_child,
            self.right_child,
            unbound=self.unbound,
            right_unbound=self.right_unbound,
            left_time_bound=self.left_time_bound,
            right_time_bound=self.right_time_bound - 1,
        )
        z_flipped = until_node._quantitative(x_flipped, normalize=normalize)
        return torch.flip(z_flipped, [3])
