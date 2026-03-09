import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, k, v):
        B, Sq, d = q.shape
        Sk = k.shape[1]
        Q = self.wq(q).view(B, Sq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(k).view(B, Sk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(v).view(B, Sk, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, Sq, d)
        return self.norm(q + self.wo(out))


class SelfAttention(nn.Module):
    def __init__(self, d, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, S, d = x.shape
        Q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, S, d)
        return self.norm(x + self.wo(out))


class DroneAttentionPolicy(nn.Module):
    """
    Latent cross-attention policy for heterogeneous node matrix.

    Input:
        x: (N,6) or (B,N,6)

    Output:
        actions: (N_d,2) or (B,N_d,2)
    """

    def __init__(
        self,
        n_feat=6,
        d=16,
        n_latent=4,
        n_heads=2,
        action_scale=2.0,
    ):
        super().__init__()

        self.d = d
        self.action_scale = float(action_scale)

        # ---- node embedding ----

        self.embed = nn.Sequential(
            nn.Linear(n_feat, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # type embedding (0 base, 1 drone, 2 grid)
        self.type_emb = nn.Embedding(3, d)

        # ---- latent tokens ----

        self.latent = nn.Parameter(
            torch.randn(n_latent, d)
        )

        # ---- attention ----

        self.cross1 = CrossAttention(d, n_heads)
        self.self1  = SelfAttention(d, n_heads)
        self.self2  = SelfAttention(d, n_heads)

        self.cross2 = CrossAttention(d, n_heads)

        # ---- action head ----

        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )

    def forward(self, x, edge_index=None):

        single = False

        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True

        B = x.shape[0]

        # ----- type ids -----

        type_id = x[..., 5].long()

        # ----- embed nodes -----

        H = self.embed(x) + self.type_emb(type_id)

        # ----- latent tokens -----

        L = self.latent.unsqueeze(0).expand(B, -1, -1)

        # node masks are identical for every batch item (fixed topology)
        dynamic_mask = type_id[0] != 2   # base + drone only  (N_b + N_d nodes)
        drone_mask   = type_id[0] == 1

        # latent reads only dynamic nodes (base + drone): 38 → 2 keys/values
        H_dyn = H[:, dynamic_mask, :]    # (B, N_b + N_d, d)

        L = self.cross1(L, H_dyn, H_dyn)
        L = self.self1(L)
        L = self.self2(L)

        H_d = H[:, drone_mask, :]          # (B, N_d, d)

        Z = self.cross2(H_d, L, L)         # (B, N_d, d)  — single batched call

        actions = self.action_scale * torch.tanh(self.head(Z))  # (B, N_d, 2)

        if single:
            return actions.squeeze(0)

        return actions
