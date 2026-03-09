# ==============================================================================
# Copyright 2020-* Luca Bortolussi
# Copyright 2020-* Laura Nenzi
# AI-CPS Group @ University of Trieste
# ==============================================================================

"""Base classes for STL/STREL formula nodes."""

from typing import Union, List, Tuple
import torch
from torch import Tensor

realnum = Union[float, int]


# =====================================================
# NODE BASE
# =====================================================

class Node:
    # Shared smooth temperature. None → hard min/max. Positive float → soft log-sum-exp.
    # As beta → ∞ the soft operators converge to hard min/max.
    _smooth_beta: float | None = None

    @classmethod
    def set_smooth_beta(cls, beta: float | None) -> None:
        """Set the global annealing temperature for all STL operators.

        Call with a positive float during training to get dense gradients;
        call with None (or after training) to restore exact STL semantics.
        """
        cls._smooth_beta = beta

    def boolean(self, x: Tensor, evaluate_at_all_times: bool = True) -> Tensor:
        z = self._boolean(x)
        return z if evaluate_at_all_times else z[:, :, :, 0]

    def quantitative(self, x: Tensor, normalize: bool = False,
                     evaluate_at_all_times: bool = True) -> Tensor:
        z = self._quantitative(x, normalize)
        return z if evaluate_at_all_times else z[:, :, :, 0]

    def _boolean(self, x: Tensor) -> Tensor: raise NotImplementedError
    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor: raise NotImplementedError


# =====================================================
# ATOM
# =====================================================

class Atom(Node):
    def __init__(self, var_index: int, threshold: realnum,
                 lte: bool = False, labels: list | None = None):
        super().__init__()
        self.var_index = var_index
        self.threshold = threshold
        self.lte = lte
        self.labels = [] if labels is None else labels  # list of accepted labels
        self._NEG_INF = -1e9

    def _mask(self, x: Tensor) -> Tensor:
        B, N, _, T = x.shape
        lab = x[:, :, -1, :]
        if self.labels:
            mask = torch.zeros_like(lab, dtype=torch.bool)
            for l in self.labels:
                mask |= (lab == l)  # mask: True for nodes with a label in self.labels
        else:
            mask = torch.ones(B, N, T, dtype=torch.bool, device=x.device)
        return mask

    def _boolean(self, x: Tensor) -> Tensor:
        xj = x[:, :, self.var_index, :].unsqueeze(2)
        z = (xj <= self.threshold) if self.lte else (xj >= self.threshold)
        return torch.where(self._mask(x).unsqueeze(2),
                           z, torch.tensor(False, device=x.device))

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        xj = x[:, :, self.var_index, :].unsqueeze(2)
        z = (-xj + self.threshold) if self.lte else (xj - self.threshold)
        NEG_INF = torch.tensor(self._NEG_INF, device=x.device, dtype=x.dtype)
        z = torch.where(self._mask(x).unsqueeze(2), z, NEG_INF)
        return torch.tanh(z) if normalize else z
