# ==============================================================================
# Copyright 2020-* Luca Bortolussi
# Copyright 2020-* Laura Nenzi
# AI-CPS Group @ University of Trieste
# ==============================================================================

"""Logical STL operators: Not, And, Or, Implies."""

import torch
from torch import Tensor

from stl.base import Node


def _align_time_horizon(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
    if left.shape[-1] == right.shape[-1]:
        return left, right
    size = min(left.shape[-1], right.shape[-1])
    return left[..., :size], right[..., :size]


# =====================================================
# LOGIC OPS
# =====================================================

class Not(Node):
    def __init__(self, child: Node): self.child = child
    def _boolean(self, x): return ~self.child._boolean(x)
    def _quantitative(self, x, normalize=False): return -self.child._quantitative(x, normalize)


class And(Node):
    def __init__(self, left: Node, right: Node): self.left, self.right = left, right
    def _boolean(self, x):
        left = self.left._boolean(x)
        right = self.right._boolean(x)
        left, right = _align_time_horizon(left, right)
        return torch.logical_and(left, right)
    def _quantitative(self, x, normalize=False):
        left = self.left._quantitative(x, normalize)
        right = self.right._quantitative(x, normalize)
        left, right = _align_time_horizon(left, right)
        beta = self._smooth_beta
        if beta is None:
            return torch.min(left, right)
        return -(1.0 / beta) * torch.logsumexp(-beta * torch.stack([left, right], dim=-1), dim=-1)


class Or(Node):
    def __init__(self, left: Node, right: Node): self.left, self.right = left, right
    def _boolean(self, x):
        left = self.left._boolean(x)
        right = self.right._boolean(x)
        left, right = _align_time_horizon(left, right)
        return torch.logical_or(left, right)
    def _quantitative(self, x, normalize=False):
        left = self.left._quantitative(x, normalize)
        right = self.right._quantitative(x, normalize)
        left, right = _align_time_horizon(left, right)
        beta = self._smooth_beta
        if beta is None:
            return torch.max(left, right)
        return (1.0 / beta) * torch.logsumexp(beta * torch.stack([left, right], dim=-1), dim=-1)


class Implies(Node):

    def __init__(self, left_child: Node, right_child: Node) -> None:
        super().__init__()
        self.left_child: Node = left_child
        self.right_child: Node = right_child
        self.implication = Or(Not(self.left_child), self.right_child)

    def _boolean(self, x):
        return self.implication._boolean(x)

    def _quantitative(self, x, normalize: bool = False):
        return self.implication._quantitative(x, normalize)
