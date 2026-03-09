from training.trainer import train_policy_gradient
import numpy as np

common = dict(
    grid_side=6, n_bases=1, n_drones=1, n_feat=6,
    T_total=70, n_iterations=30, batch_size=8, mini_batch_size=8,
    action_clip=2.0, lr=3e-4, grad_clip=1.0, action_reg_weight=1e-2,
    smooth_beta_start=1.0, smooth_beta_end=100.0, smooth_beta_anneal_iters=300,
    exploration_std_start=0.2, exploration_std_end=0.001, exploration_anneal_iters=500,
    init_pos_min=1.0, init_pos_max=6.0,
    device='cpu', show_progress=True, return_history=True,
)

print('--- MLP (hidden=128) ---')
_, h_mlp = train_policy_gradient(policy_type='mlp', policy_hidden=128, **common)
m = h_mlp['stats']['mean']
print(f'  iter 10: {np.mean(m[5:15]):.4f}  iter 30: {np.mean(m[25:]):.4f}')

print('--- EGO (hidden=32) ---')
_, h_ego = train_policy_gradient(policy_type='ego', policy_hidden=32, **common)
m = h_ego['stats']['mean']
print(f'  iter 10: {np.mean(m[5:15]):.4f}  iter 30: {np.mean(m[25:]):.4f}')
