import numpy as np
import torch
import torch.nn as nn
from policy.mlp_policy import DroneMLPPolicy

# config
GRID_SIDE=6
N_BASES=1
N_DRONES=1
N_FEAT=6
HIDDEN=128
ACTION_SCALE=2.0
POLICY_PATH='policy_1.pt'

INPUT_NAMES = ['drone_x', 'drone_y', 'drone_vx', 'drone_vy', 'drone_battery', 'base_x', 'base_y', 'base_f2', 'base_f3', 'base_occupancy']

device = torch.device('cpu')
policy = DroneMLPPolicy(n_drones=N_DRONES, n_bases=N_BASES, n_feat=N_FEAT, hidden=HIDDEN, out_channels=2, action_scale=ACTION_SCALE).to(device)
policy.load_state_dict(torch.load(POLICY_PATH, map_location=device, weights_only=True))
policy.eval()

# Weights
NEAR_ZERO_THRESHOLD = 0.01
print("=== WEIGHTS ===")
for name, lyr in policy.net.named_children():
    if isinstance(lyr, nn.Linear):
        w = lyr.weight.detach().numpy().ravel()
        b = lyr.bias.detach().numpy().ravel()
        nz_w = np.mean(np.abs(w) < NEAR_ZERO_THRESHOLD) * 100
        nz_b = np.mean(np.abs(b) < NEAR_ZERO_THRESHOLD) * 100
        print(f"Layer {name} | w_nz: {nz_w:.1f}% | b_nz: {nz_b:.1f}%")

# Saliency
N = 30
N_B = 5
xs = np.linspace(1, GRID_SIDE, N)
ys = np.linspace(1, GRID_SIDE, N)
batts = np.linspace(0.05, 1.0, N_B)
rows = []
BASE_X, BASE_Y = GRID_SIDE/2.0, GRID_SIDE/2.0
for batt in batts:
    for x in xs:
        for y in ys:
            rows.append([x, y, 0.0, 0.0, batt, BASE_X, BASE_Y, 0.0, 0.0, 0.0])
sample_inputs = torch.tensor(rows, dtype=torch.float32, device=device)

inp = sample_inputs.clone().requires_grad_(True)
out = policy.net(inp)
grad_vx = torch.autograd.grad(out[:, 0].sum(), inp, retain_graph=True)[0]
grad_vy = torch.autograd.grad(out[:, 1].sum(), inp)[0]
sal_vx = grad_vx.abs().mean(dim=0).detach().numpy()
sal_vy = grad_vy.abs().mean(dim=0).detach().numpy()

print("\n=== SALIENCY (Gradient) ===")
combined_sal = sal_vx + sal_vy
order = np.argsort(combined_sal)[::-1]
for i in order:
    print(f"{INPUT_NAMES[i]:<15} vx:{sal_vx[i]:.4f} vy:{sal_vy[i]:.4f} tot:{combined_sal[i]:.4f}")

# Occlusion
baseline_out = policy.net(sample_inputs)
baseline_mag = baseline_out.abs().mean(dim=0).detach().numpy()
occ_vx, occ_vy = [], []
for feat_idx in range(10):
    pert = sample_inputs.clone()
    pert[:, feat_idx] = 0.0
    with torch.no_grad():
        pert_out = policy.net(pert)
        pert_mag = pert_out.abs().mean(dim=0).numpy()
    occ_vx.append(abs(baseline_mag[0] - pert_mag[0]))
    occ_vy.append(abs(baseline_mag[1] - pert_mag[1]))

print("\n=== OCCLUSION ===")
occ_cmb = np.array(occ_vx) + np.array(occ_vy)
order2 = np.argsort(occ_cmb)[::-1]
for i in order2:
    print(f"{INPUT_NAMES[i]:<15} vx:{occ_vx[i]:.4f} vy:{occ_vy[i]:.4f} tot:{occ_cmb[i]:.4f}")

