"""
Instanton Finder

Compare three strategies for locating the classical instanton (kink) of
the double-well theory by minimising the Euclidean action over interior
sites (q(0)=-a and q(beta)=+a are pinned by Dirichlet BC):

  1) cnn_lbfgs  -- L-BFGS on the CNN surrogate (fast, approximate)
  2) newton     -- Newton on the exact action (slow, accurate)
  3) hybrid     -- CNN warm start, exact Newton refinement (best of both)

The narrative: the trained CNN surrogate gets close to the instanton at
near-zero cost. A few exact Newton steps then polish the result to machine
precision. This is the practical workflow that scales to expensive actions.

Note on instanton search
------------------------
For the constrained problem with Dirichlet BC, the instanton is the minimum
of the action within the kink sector, so standard minimisation suffices.
For unconstrained instanton search (true saddle-point in field configuration
space), community-standard tools are the Minimum Action Method (MAM) or
Nudged Elastic Band Method (NEBM).

Diagnostics (action and exact gradient norm) are always reported using the
exact action, regardless of which gradient drives the optimisation. Hessian
comparisons are done on interior coordinates only.

Usage
-----
Edit the CONFIGURATION block, then run:
    python instanton_finder.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import hessian as torch_hessian

from cnn_architecture import create_1d_cnn
from DoubleWell import DoubleWellPath


# ============================================================================
# CONFIGURATION  -- edit before running
# ============================================================================

# Path to a trained checkpoint produced by train.py
MODEL_PATH = Path("/your/path/here")

# Initial-guess parameters
INIT_TYPE   = "noisy_kink"   # "noisy_kink" | "wide_kink" | "step"
NOISE_SIGMA = 0.15

# Reproducibility
RANDOM_SEED = 42


# ============================================================================
# INITIAL GUESSES
# ============================================================================

def make_step_guess(N_tau: int, beta: float, a: float) -> np.ndarray:
    """Step function: q = -a for tau < beta/2, q = +a otherwise."""
    tau = np.linspace(0, beta, N_tau, endpoint=False)
    q = np.where(tau < beta / 2.0, -a, +a)
    q[0], q[-1] = -a, +a
    return q


def make_noisy_kink_guess(N_tau: int, beta: float, a: float, omega: float,
                          noise_sigma: float = 0.15) -> np.ndarray:
    """Continuum kink with Gaussian noise added at every site."""
    tau = np.linspace(0, beta, N_tau, endpoint=False)
    q = a * np.tanh(omega * (tau - beta / 2.0) / 2.0)
    q += noise_sigma * np.random.randn(N_tau)
    q[0], q[-1] = -a, +a
    return q


def make_wide_kink_guess(N_tau: int, beta: float, a: float, omega: float) -> np.ndarray:
    """Kink with width 2x the natural one -- a poor initial guess."""
    tau = np.linspace(0, beta, N_tau, endpoint=False)
    q = a * np.tanh(omega * (tau - beta / 2.0) / 4.0)
    q[0], q[-1] = -a, +a
    return q


def get_initial_guess(kind: str, path: DoubleWellPath, noise_sigma: float = 0.15) -> np.ndarray:
    """Dispatch to the requested initial-guess constructor."""
    if kind == "noisy_kink":
        return make_noisy_kink_guess(path.N_tau, path.beta, path.a, path.omega, noise_sigma)
    if kind == "wide_kink":
        return make_wide_kink_guess(path.N_tau, path.beta, path.a, path.omega)
    if kind == "step":
        return make_step_guess(path.N_tau, path.beta, path.a)
    raise ValueError(f"Unknown init kind: {kind}")


# ============================================================================
# INSTANTON FINDER
# ============================================================================

class InstantonFinder:
    """
    Locate the classical instanton via three optimisation strategies.

    All methods minimise the action over interior sites only -- boundary
    sites are pinned to +-a by Dirichlet BC. Each method returns a results
    dict with keys: path, actions, grad_norms, converged, n_steps
    (plus stage-specific keys for the hybrid method).
    """

    def __init__(self, model: torch.nn.Module, path: DoubleWellPath, std_S: float):
        self.model = model
        self.path  = path
        self.std_S = std_S
        self.model.eval()

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _exact_diagnostics(self, q: np.ndarray) -> tuple[float, float]:
        """Exact action and interior gradient norm at q (no side effects)."""
        S = self.path.get_action(q)
        g = self.path.get_gradient(q)[1:-1]
        return S, np.linalg.norm(g)

    def _assemble_full_path(self, q_int: torch.Tensor,
                            q_left: float, q_right: float) -> torch.Tensor:
        """Interior tensor (N-2,) -> full path tensor (1, 1, N) with pinned boundaries."""
        left  = q_int.new_tensor([q_left])
        right = q_int.new_tensor([q_right])
        return torch.cat([left, q_int, right]).unsqueeze(0).unsqueeze(0)

    # ----------------------------------------------------------------
    # 1. CNN surrogate (L-BFGS)
    # ----------------------------------------------------------------

    def cnn_lbfgs(
        self,
        q_init: np.ndarray,
        n_steps: int = 60,
        lr: float = 0.5,
        surrogate_tol: float = 1e-5,
        true_tol: float | None = None,
        verbose: bool = True,
    ) -> dict:
        """L-BFGS on the surrogate, with stopping based on either gradient."""
        q_left, q_right = float(q_init[0]), float(q_init[-1])
        q_int = torch.tensor(q_init[1:-1], dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [q_int], lr=lr, max_iter=20, line_search_fn="strong_wolfe",
        )

        actions, grad_norms_exact, grad_norms_surr = [], [], []
        converged = False

        for step in range(n_steps):
            def closure():
                optimizer.zero_grad()
                q_full = self._assemble_full_path(q_int, q_left, q_right)
                S_hat  = self.model(q_full).squeeze()
                S_hat.backward()
                return S_hat

            optimizer.step(closure)

            # Reconstruct full path
            q_now = np.empty_like(q_init)
            q_now[0], q_now[-1] = q_left, q_right
            q_now[1:-1] = q_int.detach().cpu().numpy()

            S_exact, gnorm_exact = self._exact_diagnostics(q_now)
            actions.append(S_exact)
            grad_norms_exact.append(gnorm_exact)

            # Surrogate gradient norm at current iterate
            q_probe      = q_int.detach().clone().requires_grad_(True)
            q_full_probe = self._assemble_full_path(q_probe, q_left, q_right)
            S_hat_probe  = self.model(q_full_probe).squeeze()
            g_hat,       = torch.autograd.grad(S_hat_probe, q_full_probe)
            gnorm_surr   = g_hat[0, 0, 1:-1].norm().item()
            grad_norms_surr.append(gnorm_surr)

            if verbose and ((step + 1) % 10 == 0 or step == 0):
                print(f"  CNN step {step+1:3d}: "
                      f"S_exact = {S_exact:.10f}, "
                      f"||grad S||_exact = {gnorm_exact:.3e}, "
                      f"||grad S_hat|| = {gnorm_surr:.3e}")

            stop_surr = (gnorm_surr  < surrogate_tol)
            stop_true = (true_tol is not None and gnorm_exact < true_tol)
            if stop_surr or stop_true:
                converged = True
                if verbose:
                    why = "true gradient" if stop_true else "surrogate gradient"
                    print(f"  CNN stopped at step {step+1} ({why} threshold reached)")
                break

        return {
            "path"                  : q_now,
            "actions"               : actions,
            "grad_norms"            : grad_norms_exact,
            "surrogate_grad_norms"  : grad_norms_surr,
            "converged"             : converged,
            "n_steps"               : step + 1,
        }

    # ----------------------------------------------------------------
    # 2. Exact Newton
    # ----------------------------------------------------------------

    def newton(
        self,
        q_init: np.ndarray,
        max_iter: int = 30,
        tol: float = 1e-10,
        # --- backtracking line search ---
        ls_alpha: float = 1e-4,    # Armijo sufficient-decrease constant
        ls_shrink: float = 0.5,    # step-size shrink factor on rejection
        ls_max_backtracks: int = 20,
        verbose: bool = True,
    ) -> dict:
        """
        Newton's method on the exact action, interior coordinates only.

        Uses Armijo backtracking line search to handle the soft translational
        zero mode of the instanton: the bare Newton step `H^{-1} grad` is
        unstable along near-zero eigenvectors of H, so we accept the full
        step only if it actually decreases the action; otherwise the step
        is shrunk by `ls_shrink` until the Armijo condition

            S(q + t*dq) <= S(q) - ls_alpha * t * grad . dq

        is satisfied (or `ls_max_backtracks` is reached, in which case the
        smallest tested step is taken).
        """
        q = q_init.copy()
        actions, grad_norms = [], []
        converged = False

        for it in range(max_iter):
            S     = self.path.get_action(q)
            grad  = self.path.get_gradient(q)[1:-1]
            H     = self.path.get_hessian(q)[1:-1, 1:-1]
            gnorm = np.linalg.norm(grad)

            actions.append(S)
            grad_norms.append(gnorm)

            if verbose and ((it + 1) % 5 == 0 or it == 0):
                print(f"  Newton step {it+1:3d}: "
                      f"S = {S:.10f}, ||grad S|| = {gnorm:.3e}")

            if gnorm < tol:
                converged = True
                if verbose:
                    print(f"  Newton converged at step {it+1}")
                break

            # Newton direction (descent on S, so we subtract dq below)
            try:
                dq = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(H, grad, rcond=None)[0]

            # Armijo backtracking: accept the full step only if S decreases
            # by at least ls_alpha * t * grad.dq. Note `dq` points uphill
            # (it's H^{-1} grad), so the descent direction is `-dq` and the
            # expected decrease is `t * grad.dq`.
            grad_dot_dq = grad @ dq
            t = 1.0
            for _ in range(ls_max_backtracks):
                q_trial = q.copy()
                q_trial[1:-1] -= t * dq
                S_trial = self.path.get_action(q_trial)
                if S_trial <= S - ls_alpha * t * grad_dot_dq:
                    break
                t *= ls_shrink

            q = q_trial

        return {
            "path"       : q,
            "actions"    : actions,
            "grad_norms" : grad_norms,
            "converged"  : converged,
            "n_steps"    : it + 1,
        }

    # ----------------------------------------------------------------
    # 3. Hybrid: CNN warm start, exact Newton refinement
    # ----------------------------------------------------------------

    def hybrid(
        self,
        q_init: np.ndarray,
        cnn_steps: int = 40,
        cnn_lr: float = 0.5,
        cnn_surrogate_tol: float = 1e-5,
        cnn_true_tol: float = 1e-4,
        newton_tol: float = 1e-10,
        verbose: bool = True,
    ) -> dict:
        if verbose:
            print("--- Stage 1: CNN surrogate ---")
        res_cnn = self.cnn_lbfgs(
            q_init, n_steps=cnn_steps, lr=cnn_lr,
            surrogate_tol=cnn_surrogate_tol, true_tol=cnn_true_tol,
            verbose=verbose,
        )

        if verbose:
            print("\n--- Stage 2: exact Newton ---")
        res_newton = self.newton(
            res_cnn["path"], max_iter=30, tol=newton_tol, verbose=verbose,
        )

        return {
            "path"           : res_newton["path"],
            "actions"        : res_cnn["actions"]    + res_newton["actions"],
            "grad_norms"     : res_cnn["grad_norms"] + res_newton["grad_norms"],
            "converged"      : res_newton["converged"],
            "n_steps_cnn"    : res_cnn["n_steps"],
            "n_steps_newton" : res_newton["n_steps"],
            "cnn_result"     : res_cnn,
            "newton_result"  : res_newton,
        }

    # ----------------------------------------------------------------
    # Hessian comparison (interior coordinates)
    # ----------------------------------------------------------------

    def interior_hessian_exact(self, q: np.ndarray) -> np.ndarray:
        """Exact action Hessian on interior coordinates."""
        H = self.path.get_hessian(q)[1:-1, 1:-1]
        return 0.5 * (H + H.T)

    def interior_hessian_cnn(self, q: np.ndarray) -> np.ndarray:
        """
        CNN Hessian on interior coordinates, in physical units.

        Uses torch.func.hessian on a closure that takes only the interior
        coordinates as a free variable, with boundaries pinned -- so the
        returned matrix is already (N-2, N-2), no slicing needed.
        """
        q_int_t = torch.tensor(q[1:-1], dtype=torch.float32)
        # Build boundary tensors outside the closure: torch.func.vmap traces
        # the closure with wrapped tensors and chokes on tensor-construction
        # calls like `q_int.new_tensor(...)` made inside.
        left  = torch.tensor([q[0]],  dtype=torch.float32)
        right = torch.tensor([q[-1]], dtype=torch.float32)

        def S_of_interior(q_int: torch.Tensor) -> torch.Tensor:
            q_full = torch.cat([left, q_int, right]).unsqueeze(0).unsqueeze(0)
            return self.model(q_full).squeeze()

        H = torch_hessian(S_of_interior)(q_int_t).detach().numpy()
        H = 0.5 * (H + H.T)
        return self.std_S * H


# ============================================================================
# PLOTTING
# ============================================================================

def plot_convergence(results: dict, S_inst: float | None = None,
                     save_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, res in results.items():
        axes[0].semilogy(res["grad_norms"], lw=1.5, label=label)
        axes[1].plot(res["actions"], lw=1.5, label=label)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel(r"$||\nabla S||_{\mathrm{interior}}$")
    axes[0].set_title("Exact gradient norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    if S_inst is not None:
        axes[1].axhline(S_inst, color="k", ls=":", lw=1, alpha=0.5,
                        label=r"$S_{\mathrm{inst}}$")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(r"$S[q]$")
    axes[1].set_title("Exact action")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_profiles(results: dict, path: DoubleWellPath,
                  save_path: str | None = None) -> None:
    tau   = np.linspace(0, path.beta, path.N_tau, endpoint=False)
    q_ref = path.classical_kink()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(tau, q_ref, "k-", lw=2, label="Reference kink")
    for label, res in results.items():
        axes[0].plot(tau, res["path"], "--", lw=1.5, label=label)
    axes[0].set_xlabel(r"$\tau$")
    axes[0].set_ylabel(r"$q(\tau)$")
    axes[0].set_title("Profiles")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    for label, res in results.items():
        axes[1].plot(tau, res["path"] - q_ref, lw=1.5, label=label)
    axes[1].axhline(0.0, color="k", lw=0.7)
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual to reference kink")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_hessian_comparison(eigs_exact: np.ndarray, eigs_cnn: np.ndarray,
                            save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    n = min(20, len(eigs_exact))
    idx = np.arange(n)

    ax.plot(idx, eigs_exact[:n], "ko-", lw=2, label="Exact")
    ax.plot(idx, eigs_cnn[:n],   "s--", lw=1.5, label="CNN")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Interior Hessian spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_cost_vs_accuracy(results: dict, path: DoubleWellPath,
                          save_path: str | None = None) -> None:
    """
    Cumulative cost (in gradient-equivalent units) vs exact gradient norm.

    Both axes are log-scaled because CNN surrogate cost (~N iters) is orders
    of magnitude smaller than Newton cost (~N_int per iter) -- on a linear
    scale the CNN line would be invisible.

    Cost model per step:
      Newton exact   = N_int  (Hessian solve)
      CNN surrogate  = 1
      Hybrid         = 1 during CNN phase, N_int during Newton phase
    """
    N_int = path.N_tau - 2
    cost_per_step_default = {
        "Newton exact" : N_int,
        "CNN surrogate": 1,
    }

    # Distinct, saturated colours -- blue / orange / green
    colours = {
        "CNN surrogate" : "#1f77b4",   # blue
        "Newton exact"  : "#d62728",   # red
        "Hybrid"        : "#2ca02c",   # green
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, res in results.items():
        gnorms = res["grad_norms"]
        if "n_steps_cnn" in res:    # hybrid CNN -> Newton
            n_cnn = res["n_steps_cnn"]
            cost  = np.cumsum([1] * n_cnn + [N_int] * (len(gnorms) - n_cnn))
        else:
            c     = cost_per_step_default.get(label, 1)
            cost  = np.arange(1, len(gnorms) + 1) * c

        ax.loglog(cost, gnorms, lw=2.0,
                  color=colours.get(label),
                  label=label)

    ax.set_xlabel("Cumulative cost (gradient-equivalent units)")
    ax.set_ylabel(r"$||\nabla S||_{\mathrm{interior}}$")
    ax.set_title("Cost vs accuracy")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def load_finder(model_path: Path) -> InstantonFinder:
    """Build an InstantonFinder from a train.py checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Set MODEL_PATH at the top of instanton_finder.py."
        )

    ckpt    = torch.load(model_path, map_location="cpu", weights_only=False)
    arch    = ckpt["architecture"]
    physics = ckpt["physics"]
    std_S   = ckpt["norm_stats"]["std_S"]

    model = create_1d_cnn(
        n_channels   = arch["n_channels"],
        kernel_sizes = arch["kernel_sizes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    path = DoubleWellPath(
        N_tau   = ckpt["N_tau"],
        beta    = physics["beta"],
        lam     = physics["lam"],
        a       = physics["a"],
        m       = physics.get("m",       1.0),
        bc      = physics.get("bc",      "dirichlet"),
        mu_bias = physics.get("mu_bias", 0.0),
        tau_pin = physics.get("tau_pin", None),
    )

    print(f"Loaded checkpoint : {model_path}")
    print(f"Architecture      : {arch}")
    print(f"N_tau={path.N_tau}, beta={path.beta}, "
          f"S_inst (cont) = {path.S_inst_continuum:.8f}")

    return InstantonFinder(model, path, std_S)


def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("=" * 60)
    print("INSTANTON FINDER")
    print("=" * 60)

    finder = load_finder(MODEL_PATH)
    path   = finder.path

    # --- Initial guess ---
    q_init    = get_initial_guess(INIT_TYPE, path, NOISE_SIGMA)
    S0, g0    = finder._exact_diagnostics(q_init)
    print(f"Initial guess     : {INIT_TYPE} (S = {S0:.8f}, ||grad S|| = {g0:.3e})")

    # ----------------------------------------------------------------
    # Run all methods
    # ----------------------------------------------------------------
    print("\n" + "=" * 60); print("METHOD 1: CNN surrogate (L-BFGS)"); print("=" * 60)
    res_cnn = finder.cnn_lbfgs(q_init.copy(), n_steps=60, lr=0.5,
                               surrogate_tol=1e-5, true_tol=1e-4)

    print("\n" + "=" * 60); print("METHOD 2: Exact Newton"); print("=" * 60)
    res_newton = finder.newton(q_init.copy(), max_iter=30, tol=1e-10)

    print("\n" + "=" * 60); print("METHOD 3: Hybrid (CNN -> Newton)"); print("=" * 60)
    res_hybrid = finder.hybrid(q_init.copy(),
                               cnn_steps=30, cnn_lr=0.5,
                               cnn_surrogate_tol=1e-5, cnn_true_tol=1e-4,
                               newton_tol=1e-10)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60); print("SUMMARY"); print("=" * 60)

    all_results = {
        "CNN surrogate" : res_cnn,
        "Newton exact"  : res_newton,
        "Hybrid"        : res_hybrid,
    }

    for label, res in all_results.items():
        Sfin, gfin = finder._exact_diagnostics(res["path"])
        if "n_steps_cnn" in res:
            steps = f"{res['n_steps_cnn']} + {res['n_steps_newton']}"
        else:
            steps = res["n_steps"]
        print(f"\n{label}:")
        print(f"  S_final            = {Sfin:.10f}")
        print(f"  ||grad S||_interior = {gfin:.3e}")
        print(f"  steps              = {steps}")
        print(f"  converged          = {res['converged']}")
        print(f"  S_final - S_inst   = {Sfin - path.S_inst_continuum:.3e}")

    # ----------------------------------------------------------------
    # Hessian comparison at Newton solution
    # ----------------------------------------------------------------
    print("\n" + "=" * 60); print("INTERIOR HESSIAN AT NEWTON SOLUTION"); print("=" * 60)

    H_exact    = finder.interior_hessian_exact(res_newton["path"])
    H_cnn      = finder.interior_hessian_cnn(res_newton["path"])
    eigs_exact = np.linalg.eigvalsh(H_exact)
    eigs_cnn   = np.linalg.eigvalsh(H_cnn)
    print(f"Exact lowest 5 : {eigs_exact[:5]}")
    print(f"CNN   lowest 5 : {eigs_cnn[:5]}")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    out_dir = MODEL_PATH.parent

    plot_convergence(all_results, S_inst=path.S_inst_continuum,
                     save_path=str(out_dir / "finder_convergence.png"))
    plot_profiles(all_results, path,
                  save_path=str(out_dir / "finder_profiles.png"))
    plot_hessian_comparison(eigs_exact, eigs_cnn,
                            save_path=str(out_dir / "finder_hessian.png"))
    plot_cost_vs_accuracy(all_results, path,
                          save_path=str(out_dir / "finder_cost_accuracy.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()