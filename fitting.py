import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import trange

def _prepare_run_curves(df, run_col="run_id", l_col="mean_radius", w_col="w_t", P_col="urban_fraction", eps=1e-12):
    """
    Return lists of arrays per run: L_runs, W_runs, P_runs (all same length per run).
    """
    L_runs, W_runs, P_runs = [], [], []
    runs = sorted(df[run_col].unique())
    for r in runs:
        d = df[df[run_col] == r]
        # ensure increasing in time (assumes index orders by time)
        d = d.sort_index()
        L = d[l_col].values.astype(float)
        W = d[w_col].values.astype(float)
        P = d[P_col].values.astype(float)
        # sanitize zeros to avoid division by zero
        P = np.maximum(P, eps)
        L = np.maximum(L, eps)
        W = np.maximum(W, 0.0)
        if len(L) < 3:
            # skip very short runs
            continue
        L_runs.append(L)
        W_runs.append(W)
        P_runs.append(P)
    return L_runs, W_runs, P_runs, runs





def _collapse_error(beta, z, L_runs, W_runs, P_runs, n_common=200):
    """
    Compute collapse error (mean squared deviation) for candidate beta,z.
    """
    # build scaled curves
    scaled_curves = []
    xmin_list, xmax_list = [], []
    for L, W, P in zip(L_runs, W_runs, P_runs):
        # scaled x and y
        x = L / (P**(1.0 / z))
        y = W / (P**beta)
        # keep only finite
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if mask.sum() < 3:
            return np.inf  # not enough points
        x, y = x[mask], y[mask]
        xmin_list.append(x.min()); xmax_list.append(x.max())
        scaled_curves.append((x, y))

    # find common interval across all runs
    x_min = max(xmin_list)
    x_max = min(xmax_list)
    if not (x_max > x_min):
        return np.inf

    # common grid in log-space
    x_common = np.exp(np.linspace(np.log(x_min), np.log(x_max), n_common))

    # interpolate and compute mean-square scatter around mean curve
    interp_vals = []
    for x, y in scaled_curves:
        f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
        interp_vals.append(f(x_common))
    C = np.vstack(interp_vals)  # shape (n_runs, n_common)
    mean_curve = np.nanmean(C, axis=0)
    # squared deviations
    sq = (C - mean_curve[None, :])**2
    # use nanmean in case of extrapolation producing NaNs
    mse = np.nanmean(sq)
    return mse




def find_best_collapse(df,
                       run_col="run_id",
                       l_col="mean_radius",
                       w_col="w_t",
                       P_col="urban_fraction",
                       beta_range=(-1.0, 2.0),
                       z_range=(0.1, 4.0),
                       beta_steps=101,
                       z_steps=121,
                       n_common=200,
                       bootstrap_iters=0,
                       verbose=True):
    """
    Grid-search best (beta, z) that minimizes collapse thickness.

    Returns:
      result dict with best beta,z, error arrays, and optionally bootstrap stats.
    """
    L_runs, W_runs, P_runs, runs = _prepare_run_curves(df, run_col, l_col, w_col, P_col)
    if len(L_runs) == 0:
        raise ValueError("No valid runs found in dataframe.")

    beta_vals = np.linspace(beta_range[0], beta_range[1], beta_steps)
    z_vals = np.linspace(z_range[0], z_range[1], z_steps)

    errors = np.full((len(beta_vals), len(z_vals)), np.inf, dtype=float)

    # grid search
    iterator = trange(len(beta_vals), desc="beta grid") if verbose else range(len(beta_vals))
    for i in iterator:
        b = beta_vals[i]
        for j in range(len(z_vals)):
            z = z_vals[j]
            err = _collapse_error(b, z, L_runs, W_runs, P_runs, n_common=n_common)
            errors[i, j] = err

    # pick best
    idx = np.unravel_index(np.nanargmin(errors), errors.shape)
    best_beta = beta_vals[idx[0]]
    best_z = z_vals[idx[1]]
    best_error = errors[idx]

    result = {
        "best_beta": float(best_beta),
        "best_z": float(best_z),
        "best_error": float(best_error),
        "beta_vals": beta_vals,
        "z_vals": z_vals,
        "errors": errors,
        "runs": runs
    }

    # optional bootstrap: resample runs with replacement and refit
    if bootstrap_iters and bootstrap_iters > 0:
        zs_boot = []
        betas_boot = []
        n_runs = len(L_runs)
        for b_iter in range(bootstrap_iters):
            idxs = np.random.choice(n_runs, n_runs, replace=True)
            Lb = [L_runs[k] for k in idxs]
            Wb = [W_runs[k] for k in idxs]
            Pb = [P_runs[k] for k in idxs]
            # coarse grid for bootstrap (to speed up)
            err_min = np.inf
            b_best = None; z_best = None
            for bval in beta_vals:
                for zval in z_vals:
                    err = _collapse_error(bval, zval, Lb, Wb, Pb, n_common=n_common)
                    if err < err_min:
                        err_min = err; b_best = bval; z_best = zval
            betas_boot.append(b_best); zs_boot.append(z_best)
        result["bootstrap_betas"] = np.array(betas_boot)
        result["bootstrap_zs"] = np.array(zs_boot)
        result["bootstrap_beta_mean"] = float(np.mean(betas_boot))
        result["bootstrap_beta_std"] = float(np.std(betas_boot))
        result["bootstrap_z_mean"] = float(np.mean(zs_boot))
        result["bootstrap_z_std"] = float(np.std(zs_boot))

    return result

def plot_collapse(df, result,
                  run_col="run_id",
                  l_col="mean_radius",
                  w_col="w_t",
                  P_col="urban_fraction",
                  beta=None, z=None,
                  plot_mean=True, alpha=0.3, figsize=(8,6)):
    """
    Plot the collapsed curves using provided beta,z (if None uses result['best_*']).
    """
    if beta is None: beta = result["best_beta"]
    if z is None: z = result["best_z"]

    L_runs, W_runs, P_runs, runs = _prepare_run_curves(df, run_col, l_col, w_col, P_col)
    plt.figure(figsize=figsize)
    scaled = []
    for L, W, P in zip(L_runs, W_runs, P_runs):
        x = L / (P**(1.0 / z))
        y = W / (P**beta)
        plt.plot(x, y, color="C0", alpha=alpha)
        scaled.append((x,y))
    if plot_mean:
        # compute mean over interpolated common grid
        xmins = [x.min() for x,_ in scaled]; xmaxs = [x.max() for x,_ in scaled]
        x_min = max(xmins); x_max = min(xmaxs)
        if x_max > x_min:
            x_common = np.exp(np.linspace(np.log(x_min), np.log(x_max), 300))
            Ys = []
            for x,y in scaled:
                f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
                Ys.append(f(x_common))
            C = np.vstack(Ys)
            mean_curve = np.nanmean(C, axis=0)
            plt.plot(x_common, mean_curve, color="k", lw=2, label=f"mean (beta={beta:.3f}, z={z:.3f})")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("l / P^(1/z)")
    plt.ylabel("w / P^beta")
    plt.legend()
    plt.title("Curve collapse")
    plt.tight_layout()
    plt.show()
