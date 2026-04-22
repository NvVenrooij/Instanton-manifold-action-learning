import numpy as np
import matplotlib.pyplot as plt
import os, json
from datetime import datetime
from DoubleWell import DoubleWellPath

# ============================================================
# Parameters
# ============================================================
# In params dict, add:

params = dict(
    # --- Lattice & physics ---
    N_tau=128,         # number of lattice sites in imaginary time
    beta=10.0,         # total imaginary-time extent (inverse temperature); eps = beta/N_tau
    lam=1.0,           # coupling in V(q) = lam * (q^2 - a^2)^2
    a=1.0,             # vacuum expectation value; minima of V at q = +-a
    m=1.0,             # mass (kinetic-term coefficient)
    bc='dirichlet',    # boundary conditions: 'dirichlet' (q_0=-a, q_N=+a) or 'periodic'

    # --- Dataset size ---
    n_configs=2000,    # total number of configurations to generate
    n_centers=100,     # number of distinct kink centres tau0 to sample clouds around

    # --- Mean-field cloud sampling ---
    max_mode=10,            # number of low-frequency modes used to perturb the background
    uv_sigma=0.1,           # std of additional per-site Gaussian noise (adds UV content)
    DeltaS_target=5.0,      # target excess action above the classical kink per sample
    DeltaS_max_factor=30.0, # reject samples with |DeltaS - target| > factor * std(DeltaS)

    # --- Symmetry-breaking bias (pins the instanton centre) ---
    mu_bias=0.0001,    # strength of quadratic bias weighting q^2 by ((tau - tau_pin)/beta)^2
    tau_pin=5,         # tau value where the bias vanishes (typically beta/2)

    # --- Instanton-centre distribution ---
    tau0_frac_min=0.2, # lower bound on kink centres as a fraction of beta
    tau0_frac_max=0.8, # upper bound on kink centres as a fraction of beta
)

p = params
path = DoubleWellPath(p['N_tau'], p['beta'], p['lam'], p['a'],
                      m=p['m'], bc=p['bc'],
                      mu_bias=p['mu_bias'], tau_pin=p['tau_pin'])

print("Double-well instanton cloud generator")
print(f"  N_tau = {p['N_tau']}, beta = {p['beta']}, eps = {path.eps:.4f}")
print(f"  lam = {p['lam']}, a = {p['a']}, omega = {path.omega:.4f}")
print(f"  S_inst (continuum) = {path.S_inst_continuum:.4f}")
print(f"  BC: {p['bc']}")
print(f"  n_centers = {p['n_centers']}, max_mode = {p['max_mode']}")
print(f"  DeltaS_target = {p['DeltaS_target']}, uv_sigma = {p['uv_sigma']}")

# ============================================================
# Generate instanton clouds
# ============================================================
tau_grid = np.linspace(0, p['beta'], p['N_tau'], endpoint=False)
tau0_list = np.linspace(
    p['tau0_frac_min'] * p['beta'],
    p['tau0_frac_max'] * p['beta'],
    p['n_centers'],
)

samples_per_center = p['n_configs'] // p['n_centers']
remainder = p['n_configs'] % p['n_centers']

all_cfgs, all_actions, all_grads = [], [], []
all_charges, all_centers, all_center_labels = [], [], []
exact_center_actions = []

print("\nGenerating instanton clouds...\n")

for ic, tau0 in enumerate(tau0_list):
    n_local = samples_per_center + (1 if ic < remainder else 0)
    center = path.classical_kink(tau0)
    S_center = path.get_action(center)
    exact_center_actions.append(S_center)

    cfgs, actions = path.mean_field_cloud(
        n_samples=n_local, max_mode=p['max_mode'],
        uv_sigma=p['uv_sigma'], DeltaS_target=p['DeltaS_target'],
        DeltaS_max_factor=p['DeltaS_max_factor'],
        center=center, S_center=S_center, include_center=True,
    )

    grads = np.array([path.get_gradient(q) for q in cfgs])
    charges = np.array([path.topological_charge(q) for q in cfgs])
    centers = np.array([path.find_zero_crossing(q) for q in cfgs])

    all_cfgs.append(cfgs)
    all_actions.append(actions)
    all_grads.append(grads)
    all_charges.append(charges)
    all_centers.append(centers)
    all_center_labels.append(np.full(len(cfgs), tau0))

cfgs = np.concatenate(all_cfgs)
actions = np.concatenate(all_actions)
gradients = np.concatenate(all_grads)
charges = np.concatenate(all_charges)
centers = np.concatenate(all_centers)
center_labels = np.concatenate(all_center_labels)
exact_center_actions = np.array(exact_center_actions)

print(f"Generated {len(cfgs)} instanton-cloud samples")
print(f"Action: {actions.mean():.4f} ± {actions.std():.4f}")
print(f"Charge: {charges.mean():.4f} ± {charges.std():.4f}")
print(f"Valid zero crossings: {np.sum(~np.isnan(centers))}/{len(centers)}")

# ============================================================
# Save
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("data", f"instanton_cloud_N{p['N_tau']}_b{p['beta']}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

for name, arr in [('configs', cfgs), ('actions', actions), ('gradients', gradients),
                  ('charges', charges), ('centers', centers), ('center_labels', center_labels)]:
    np.save(os.path.join(run_dir, f"{name}.npy"), arr)

with open(os.path.join(run_dir, "params.json"), 'w') as f:
    json.dump(params, f, indent=2)

# ============================================================
# Derived quantities for diagnostics
# ============================================================
grad_norms_inner = np.linalg.norm(gradients[:, 1:-1], axis=1)

# action above the classical kink each sample was generated from
deltaS = np.array([
    path.get_action(q) - path.get_action(path.classical_kink(tau0))
    for q, tau0 in zip(cfgs, center_labels)
])

# distance to parent classical kink
dist_to_center = np.sqrt(np.mean(
    (cfgs - np.array([path.classical_kink(tau0) for tau0 in center_labels]))**2,
    axis=1,
))

# ============================================================
# Diagnostic plots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# (1) exact center actions vs continuum
axes[0, 0].plot(tau0_list, exact_center_actions, 'o-', ms=3, label='discrete kink')
axes[0, 0].axhline(path.S_inst_continuum, color='r', ls='--', label='continuum $S_{\\rm inst}$')
axes[0, 0].set(xlabel=r'$\tau_0$', ylabel=r'$S[q_{\rm cl}]$', title='Action of exact kink centers')
axes[0, 0].legend()

# (2) instanton center distribution
valid_centers = centers[~np.isnan(centers)]
axes[0, 1].hist(valid_centers, bins=30, alpha=0.8, label='sampled')
axes[0, 1].hist(center_labels, bins=20, alpha=0.5, label='target')
axes[0, 1].set(xlabel=r'$\tau_0$', ylabel='Count', title='Instanton center distribution')
axes[0, 1].legend()

# (3) action distribution
axes[0, 2].hist(actions, bins=30, alpha=0.8)
axes[0, 2].axvline(path.S_inst_continuum, color='r', ls='--', label='continuum $S_{\\rm inst}$')
axes[0, 2].axvline(exact_center_actions.mean(), color='g', ls='--', label='mean discrete center')
axes[0, 2].set(xlabel='S[q]', ylabel='Count', title='Action distribution')
axes[0, 2].legend()

# (4) sample paths
for _ in range(40):
    j = np.random.randint(len(cfgs))
    axes[1, 0].plot(tau_grid, cfgs[j], alpha=0.2, lw=0.8)
axes[1, 0].plot(tau_grid, path.classical_kink(), 'k--', lw=2, label='classical kink')
axes[1, 0].set(xlabel=r'$\tau$', ylabel=r'$q(\tau)$', title='Sampled paths')
axes[1, 0].legend()

# (5) topological charge
axes[1, 1].hist(charges, bins=30, alpha=0.8)
axes[1, 1].axvline(1.0, color='r', ls='--', label='Q=1')
axes[1, 1].set(xlabel='Q', ylabel='Count', title='Topological charge')
axes[1, 1].legend()

# (6) ΔS distribution
axes[1, 2].hist(deltaS, bins=30, alpha=0.8)
axes[1, 2].axvline(p['DeltaS_target'], color='r', ls='--', label='target')
axes[1, 2].set(xlabel=r'$\Delta S$', ylabel='Count', title=r'$\Delta S = S[q] - S[q_{\rm cl}]$')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "diagnostics.png"), dpi=160)
plt.show()

print(f"\nSaved to {run_dir}/")