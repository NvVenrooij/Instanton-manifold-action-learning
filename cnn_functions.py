"""
Training and Evaluation Utilities for the 1D Action CNN (Double Well)

Provides:
- train_model        : main training loop with optional gradient matching
                       and saddle-point pinning. Z₂ augmentation (q → -q)
                       is applied per-batch when lambda_grad > 0.
- saddle_pinning_loss: auxiliary loss penalizing ||dS_θ/dq||² at classical
                       instanton configurations.
- evaluate_model     : action prediction metrics on a held-out set.
- plot_training_history, plot_evaluation_results : diagnostic figures.
- plot_hessian_check     : Hessian eigenvalue comparison at the instanton.
- plot_gradient_check    : per-sample gradient comparison on validation paths.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


# ============================================================================
# AUGMENTATION
# ============================================================================

def _random_z2_flip(
    q_batch: torch.Tensor,
    grad_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Z₂ augmentation: q → -q with 50% probability per sample.

    The double-well action satisfies S[-q] = S[q], so flipping is an exact
    symmetry of the target. The gradient transforms as dS/dq → -dS/dq.

    Parameters
    ----------
    q_batch : (B, 1, N_tau)
    grad_batch : (B, N_tau) or None

    Returns
    -------
    q_aug : (B, 1, N_tau)
    grad_aug : (B, N_tau) or None
    """
    B = q_batch.shape[0]
    sign = 1.0 - 2.0 * (torch.rand(B) < 0.5).float()   # ±1

    q_aug = q_batch * sign[:, None, None]

    if grad_batch is not None:
        return q_aug, grad_batch * sign[:, None]

    return q_aug, None


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model: nn.Module,
    train_q: torch.Tensor,
    train_S: torch.Tensor,
    val_q: torch.Tensor,
    val_S: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    # -------- gradient matching --------
    lambda_grad: float = 0.0,
    train_grad_S: torch.Tensor | None = None,
    val_grad_S: torch.Tensor | None = None,
    use_z2_augmentation: bool = True,
    # -------- saddle-point pinning --------
    lambda_saddle: float = 0.0,
    saddle_configs: list[torch.Tensor] | None = None,
    # -------- checkpointing --------
    checkpoint_path=None,
    checkpoint_every: int = 0,
    verbose: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    Train the 1D action CNN.

    Loss
    ----
    L = L_S  +  lambda_grad * L_grad  +  lambda_saddle * L_saddle

    where:
      L_S     = MSE between predicted and target (normalized) actions.
      L_grad  = MSE between predicted and exact dS/dq_i (gradient matching).
      L_saddle= mean ||dS_θ/dq||² evaluated at classical instanton configs
                (saddle-point pinning; see saddle_pinning_loss).

    Z₂ augmentation (q → -q) is applied per-batch when gradient matching
    is active (use_z2_augmentation=True). The action is invariant and the
    gradient sign flips consistently, so this doubles effective dataset size
    at no extra simulation cost.

    Parameters
    ----------
    train_q, val_q : (N, 1, N_tau)
        Path configurations.
    train_S, val_S : (N,)
        Normalized action values.
    lambda_grad : float
        Weight for gradient matching loss. Set to 0 to disable.
    train_grad_S : (N, N_tau) or None
        Exact dS/dq_i values. Required when lambda_grad > 0.
    use_z2_augmentation : bool
        Apply Z₂ flip augmentation during gradient-matching batches.
        Only effective when lambda_grad > 0.
    lambda_saddle : float
        Weight for saddle-point pinning. Set to 0 to disable.
    saddle_configs : list of (1, 1, N_tau) tensors or None
        Classical instanton profiles for saddle pinning.
    checkpoint_path : str or None
        If given, save a checkpoint here every checkpoint_every epochs.
    checkpoint_every : int
        Checkpoint interval in epochs. Ignored if checkpoint_path is None.
    verbose : bool
        Print per-epoch training statistics.

    Returns
    -------
    train_losses, val_losses : list of float
        Per-epoch average total loss (train) and MSE loss (val).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses: List[float] = []
    val_losses: List[float] = []

    n_batches = (len(train_q) + batch_size - 1) // batch_size

    use_grad   = (lambda_grad   > 0.0 and train_grad_S is not None)
    use_val_grad = (val_grad_S is not None)
    use_saddle = (lambda_saddle > 0.0 and saddle_configs is not None)
    augment    = use_grad and use_z2_augmentation

    if verbose:
        print(
            f"  Epochs: {n_epochs}  |  Batch: {batch_size}  |  "
            f"Batches/epoch: {n_batches}  |  LR: {learning_rate}"
        )
        print(
            f"  lambda_grad = {lambda_grad} ({'ON' if use_grad else 'OFF'})  |  "
            f"Z2 aug = {'ON' if augment else 'OFF'}  |  "
            f"lambda_saddle = {lambda_saddle} ({'ON' if use_saddle else 'OFF'})"
        )
        print()

    for epoch in range(n_epochs):
        model.train()

        cum_LS = cum_Lgrad = cum_Lsaddle = cum_total = 0.0

        indices = torch.randperm(len(train_q))

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end   = min(start + batch_size, len(train_q))
            idx   = indices[start:end]

            # ---------------------------
            # BASE ACTION LOSS
            # ---------------------------
            q_batch = train_q[idx]      # (B, 1, N_tau)
            S_true  = train_S[idx]      # (B,)

            S_pred = model(q_batch).squeeze()
            L_S    = criterion(S_pred, S_true)
            loss   = L_S
            L_grad = L_saddle = None

            # ---------------------------
            # GRADIENT MATCHING
            # ---------------------------
            if use_grad:
                q_leaf     = train_q[idx].detach().clone()
                grad_true  = train_grad_S[idx].clone()

                if augment:
                    q_leaf, grad_true = _random_z2_flip(q_leaf, grad_true)

                q_leaf.requires_grad_(True)
                S_pred_g = model(q_leaf).squeeze()
                grad_pred, = torch.autograd.grad(
                    S_pred_g.sum(), q_leaf, create_graph=True
                )
                grad_pred = grad_pred.squeeze(1)    # (B, N_tau)

                L_grad = criterion(grad_pred, grad_true)
                loss   = loss + lambda_grad * L_grad

            # ---------------------------
            # SADDLE-POINT PINNING
            # ---------------------------
            if use_saddle:
                L_saddle = saddle_pinning_loss(model, saddle_configs)
                loss     = loss + lambda_saddle * L_saddle

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_LS     += L_S.item()
            cum_Lgrad  += L_grad.item()   if L_grad   is not None else 0.0
            cum_Lsaddle+= L_saddle.item() if L_saddle is not None else 0.0
            cum_total  += loss.item()

        avg_LS      = cum_LS      / n_batches
        avg_Lgrad   = cum_Lgrad   / n_batches
        avg_Lsaddle = cum_Lsaddle / n_batches
        avg_total   = cum_total   / n_batches

        train_losses.append(avg_total)

        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()

        with torch.no_grad():
            val_pred = model(val_q).squeeze()
            val_loss = criterion(val_pred, val_S).item()
        val_losses.append(val_loss)

        val_grad_loss = None
        if use_val_grad:
            q_val_leaf = val_q.detach().clone().requires_grad_(True)
            S_val_pred = model(q_val_leaf).squeeze()
            grad_val_pred, = torch.autograd.grad(S_val_pred.sum(), q_val_leaf)
            grad_val_pred  = grad_val_pred.squeeze(1)
            val_grad_loss  = criterion(grad_val_pred, val_grad_S).item()

        # ---------------------------
        # LOGGING
        # ---------------------------
        if verbose and (epoch + 1) % 10 == 0:
            train_parts = [f"L_S={avg_LS:.3e}"]
            val_parts   = [f"L_S={val_loss:.3e}"]
            if use_grad:
                train_parts.append(f"L_grad={avg_Lgrad:.3e}")
                if val_grad_loss is not None:
                    val_parts.append(f"L_grad={val_grad_loss:.3e}")
            if use_saddle:
                train_parts.append(f"L_saddle={avg_Lsaddle:.3e}")

            print(
                f"Epoch {epoch+1:3d}/{n_epochs}  |  "
                f"train  {'  '.join(train_parts)}  |  "
                f"val  {'  '.join(val_parts)}"
            )

        # ---------------------------
        # CHECKPOINT
        # ---------------------------
        if (
            checkpoint_path is not None
            and checkpoint_every > 0
            and (epoch + 1) % checkpoint_every == 0
        ):
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch + 1,
                    "train_losses":     train_losses,
                    "val_losses":       val_losses,
                },
                checkpoint_path,
            )

    if verbose:
        print("\n  Training complete.\n")

    return train_losses, val_losses


# ============================================================================
# SADDLE-POINT PINNING
# ============================================================================

def saddle_pinning_loss(
    model: nn.Module,
    saddle_configs: list[torch.Tensor],
) -> torch.Tensor:
    """
    Penalize nonzero action gradient at classical instanton configurations.

        L_saddle = (1/n) Σ_configs  mean( (dS_θ/dq)² )  [interior sites only]

    The classical instanton is a saddle point of the action functional, so
    dS/dq = 0 there (up to boundary terms). This loss pins the surrogate to
    reproduce that exact stationarity condition — constraining it beyond what
    the action-value MSE alone can enforce.

    Parameters
    ----------
    model : nn.Module
        The action surrogate.
    saddle_configs : list of (1, 1, N_tau) tensors
        Classical instanton profiles q_cl(τ) at various centre positions τ₀.
    """
    total = torch.tensor(0.0)
    for q_cl in saddle_configs:
        q = q_cl.detach().clone().requires_grad_(True)
        S = model(q).squeeze()
        grad_q, = torch.autograd.grad(S, q, create_graph=True)
        total = total + (grad_q[..., 1:-1] ** 2).mean()

    return total / len(saddle_configs)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_q: torch.Tensor,
    test_S: torch.Tensor,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the trained model on a held-out set.

    Parameters
    ----------
    test_q : (N, 1, N_tau)
    test_S : (N,)  normalized action values.

    Returns
    -------
    dict with keys: predictions, true_actions, mse, mae,
                    rel_mae_pct, mean_abs_pct_error, errors, pct_errors.
    """
    model.eval()

    with torch.no_grad():
        predictions = model(test_q).squeeze().detach().cpu().numpy()

    true_actions = test_S.detach().cpu().numpy()
    errors       = predictions - true_actions
    mse          = np.mean(errors ** 2)
    mae          = np.mean(np.abs(errors))
    true_std     = np.std(true_actions)
    rel_mae_pct  = 100.0 * mae / (true_std + 1e-12)

    min_denom         = 0.05 * true_std
    denom             = np.maximum(np.abs(true_actions), min_denom)
    pct_errors        = 100.0 * errors / denom
    mean_abs_pct_error= np.mean(np.abs(pct_errors))

    if verbose:
        print(f"\n{'='*60}")
        print("Test Results")
        print(f"{'='*60}")
        print(f"MSE:                    {mse:.6f}")
        print(f"MAE:                    {mae:.6f}")
        print(f"Rel MAE (vs std):       {rel_mae_pct:.2f}%")
        print(f"Mean |pct error|*:      {mean_abs_pct_error:.2f}%")
        print(f"  (*denom >= {min_denom:.3f} in normalized units)")
        print(f"{'='*60}\n")

    return {
        "predictions":        predictions,
        "true_actions":       true_actions,
        "mse":                mse,
        "mae":                mae,
        "rel_mae_pct":        rel_mae_pct,
        "mean_abs_pct_error": mean_abs_pct_error,
        "errors":             errors,
        "pct_errors":         pct_errors,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = "training_history.png",
) -> None:
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss",   linewidth=2)
    plt.plot(val_losses,   label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss",  fontsize=12)
    plt.title("Training History", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training history to '{save_path}'")
    plt.show()
    plt.close()


def plot_evaluation_results(
    results: dict,
    save_path: str = "evaluation_results.png",
) -> None:
    """Three-panel evaluation figure: predicted vs true, error histogram, sample traces."""
    predictions  = results["predictions"]
    true_actions = results["true_actions"]
    errors       = results["errors"]
    mae          = results["mae"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Predicted vs True
    ax = axes[0]
    ax.scatter(true_actions, predictions, alpha=0.5, s=10)
    lo, hi = min(true_actions.min(), predictions.min()), max(true_actions.max(), predictions.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect")
    ax.set_xlabel("True Action")
    ax.set_ylabel("Predicted Action")
    ax.set_title("Predicted vs True")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Error distribution
    ax = axes[1]
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (MAE = {mae:.4f})")
    ax.grid(True, alpha=0.3)

    # 3. Sample traces
    ax = axes[2]
    n  = min(200, len(predictions))
    ax.plot(predictions[:n],  label="Predicted", alpha=0.7, lw=1.5)
    ax.plot(true_actions[:n], label="True",      alpha=0.7, lw=1.5)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Action")
    ax.set_title("Sample Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved evaluation results to '{save_path}'")
    plt.show()
    plt.close()


def plot_hessian_check(
    eigs_exact: np.ndarray,
    eigs_cnn: np.ndarray,
    save_path: str = "hessian_check.png",
) -> None:
    """
    Two-panel Hessian eigenvalue comparison at the classical instanton.

    Left : eigenvalue index vs value for exact and CNN Hessians (lowest 20).
    Right: parity plot — exact eigenvalue vs CNN eigenvalue.

    The lowest mode is expected to be near-zero (soft translational mode).
    Good surrogate training should reproduce the low-lying spectrum closely.

    Parameters
    ----------
    eigs_exact : (N_tau,) sorted eigenvalues of the exact Hessian.
    eigs_cnn   : (N_tau,) sorted eigenvalues of the CNN Hessian (physical units).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(eigs_exact[:20], "o-",  label="Exact")
    axes[0].plot(eigs_cnn[:20],   "s--", label="CNN")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Hessian eigenvalues (lowest 20)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    lim = max(eigs_exact.max(), eigs_cnn.max())
    axes[1].scatter(eigs_exact, eigs_cnn, s=10, alpha=0.5)
    axes[1].plot([0, lim], [0, lim], "r--", lw=1.5)
    axes[1].set_xlabel("Exact eigenvalue")
    axes[1].set_ylabel("CNN eigenvalue")
    axes[1].set_title("Hessian eigenvalue parity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved Hessian check to '{save_path}'")
    plt.show()
    plt.close()


def plot_gradient_check(
    tau_grid: np.ndarray,
    val_grads_true: np.ndarray,
    val_grads_pred: np.ndarray,
    val_configs: np.ndarray,
    n_panels: int = 6,
    save_path: str = "gradient_check.png",
) -> None:
    """
    Compare exact and CNN-predicted gradients on random validation paths.

    Parameters
    ----------
    tau_grid       : (N_tau,) Euclidean time grid.
    val_grads_true : (N_val, N_tau) exact dS/dq on validation set.
    val_grads_pred : (N_val, N_tau) CNN-predicted dS/dq on validation set.
    val_configs    : (N_val, N_tau) validation path configurations (for indexing).
    n_panels       : number of random samples to show (arranged in 2 rows).
    """
    n_cols = (n_panels + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 7))

    for i, ax in enumerate(axes.flat):
        if i >= n_panels:
            ax.axis("off")
            continue
        j = np.random.randint(len(val_configs))
        ax.plot(tau_grid, val_grads_true[j], label="Exact", lw=1.5)
        ax.plot(tau_grid, val_grads_pred[j], "--", label="CNN", lw=1.5)
        ax.set_title(f"Sample {j}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Gradient comparison on validation paths", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved gradient check to '{save_path}'")
    plt.show()
    plt.close()