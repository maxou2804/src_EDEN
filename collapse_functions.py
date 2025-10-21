import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def collapse_error_normalized(exponents, x_list, y_list, P_list, n_interp=200, min_curves=2, eps=1e-12):
    """
    Normalized collapse error:
      - use intersection of rescaled x ranges across curves
      - interpolate on log-spaced grid inside intersection
      - at each x compute std(y)/mean(y) across curves (relative scatter)
      - weight by number of valid curves at that x (so points with many overlapping curves matter more)
    """
    beta, inv_z = exponents
    M = len(x_list)
    # compute rescaled arrays and find intersection [x_min, x_max]
    x_rescaled_list = []
    y_rescaled_list = []
    x_mins, x_maxs = [], []
    for x, y, P in zip(x_list, y_list, P_list):
        # ensure arrays are numpy, sorted by x
        x = np.asarray(x)
        y = np.asarray(y)
        order = np.argsort(x)
        x = x[order]; y = y[order]
        x_r = x / (P**inv_z)
        y_r = y / (P**beta)
        x_rescaled_list.append(x_r)
        y_rescaled_list.append(y_r)
        x_mins.append(np.nanmin(x_r))
        x_maxs.append(np.nanmax(x_r))

    # intersection domain
    x_min = max(x_mins)
    x_max = min(x_maxs)
    if x_max <= x_min:
        return np.inf  # no overlap — reject

    # use a log-spaced or linear grid depending on range
    if x_min <= 0:
        # shift slightly to positive if needed (should rarely happen)
        x_min = max(x_min, 1e-12)
    x_common = np.exp(np.linspace(np.log(x_min), np.log(x_max), n_interp))

    # interpolate each curve on x_common
    Y = np.full((M, n_interp), np.nan, dtype=float)
    valid_counts = np.zeros(n_interp, dtype=int)
    for i, (xr, yr) in enumerate(zip(x_rescaled_list, y_rescaled_list)):
        f = interp1d(xr, yr, bounds_error=False, fill_value=np.nan)
        yi = f(x_common)
        Y[i, :] = yi
        valid_counts += ~np.isnan(yi)

    # At each x, compute relative std = std/mean with robust handling
    rel_std = np.full(n_interp, np.nan, dtype=float)
    for k in range(n_interp):
        vals = Y[:, k]
        vals = vals[~np.isnan(vals)]
        nval = vals.size
        if nval >= min_curves:
            mu = np.mean(vals)
            sigma = np.std(vals, ddof=0)
            # avoid division by zero: if mean ~ 0 use abs mean of vals
            denom = max(abs(mu), eps)
            rel_std[k] = sigma / denom
        else:
            rel_std[k] = np.nan

    # weight by number of valid curves at that x (or uniform if you prefer)
    weights = valid_counts.astype(float)
    weights[np.isnan(rel_std)] = 0.0
    if weights.sum() == 0:
        return np.inf
    # compute weighted mean of rel_std (ignoring nan positions)
    mask = ~np.isnan(rel_std)
    weighted_mean_rel_std = np.sum(rel_std[mask] * weights[mask]) / np.sum(weights[mask])
    return weighted_mean_rel_std


def minimize_collapse_normalized(x_list, y_list, P_list,
                                 beta_guess=0.33, inv_z_guess=0.66,
                                 bounds=(( -2, 2), (0.0, 2.0)),
                                 n_interp=200):
    """
    Minimize normalized collapse error with bounds and return result dict.
    """
    initial = np.array([beta_guess, inv_z_guess], dtype=float)

    def objective(v):
        return collapse_error_normalized(v, x_list, y_list, P_list, n_interp=n_interp)

    res = minimize(objective, x0=initial, method='L-BFGS-B', bounds=bounds,
                   options={'ftol':1e-10, 'maxiter': 1000, 'disp': False})
    out = {
        "beta": float(res.x[0]),
        "inv_z": float(res.x[1]),
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": res.message,
        "nfev": res.nfev
    }
    return out


def plot_before_after(x_list, y_list, P_list, beta, inv_z, x_label="x", y_label="y", logscale=True):
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1,2,1)
    for x, y in zip(x_list, y_list):
        ax1.plot(x, y, alpha=0.6)
    ax1.set_title("Before rescaling")
    ax1.set_xlabel(x_label); ax1.set_ylabel(y_label)
    if logscale:
        ax1.set_xscale('log'); ax1.set_yscale('log')

    ax2 = plt.subplot(1,2,2)
    for x, y, P in zip(x_list, y_list, P_list):
        xr = x / (P**inv_z)
        yr = y / (P**beta)
        ax2.plot(xr, yr, alpha=0.6)
    ax2.set_title(f"After rescaling (beta={beta:.3f}, 1/z={inv_z:.3f})")
    ax2.set_xlabel("x / P^(1/z)"); ax2.set_ylabel("y / P^beta")
    if logscale:
        ax2.set_xscale('log'); ax2.set_yscale('log')
    plt.tight_layout()
    plt.show()


def grid_error_map(x_list, y_list, P_list, beta_vals, inv_z_vals, n_interp=200):
    """
    Compute error for grid of beta and inv_z to inspect error landscape.
    Returns 2D errors array with shape (len(beta_vals), len(inv_z_vals)).
    """
    E = np.full((len(beta_vals), len(inv_z_vals)), np.nan)
    for i, b in enumerate(beta_vals):
        for j, iz in enumerate(inv_z_vals):
            E[i, j] = collapse_error_normalized((b, iz), x_list, y_list, P_list, n_interp=n_interp)
    return E





def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

import numpy as np

def add_array_with_padding(arrays, new_array):
    """
    Add a new 1D array to a list of arrays, padding all arrays with NaN
    so they all have equal length.

    Parameters
    ----------
    arrays : list of np.ndarray
        The list of existing 1D arrays.
    new_array : np.ndarray or list
        The new array to add.

    Returns
    -------
    arrays : list of np.ndarray
        Updated list of arrays with consistent lengths.
    """
    new_array = np.array(new_array, dtype=float)  # ensure float for NaN support
    new_len = len(new_array)

    # Determine current maximum length among existing arrays
    if arrays:
        max_len = max(len(a) for a in arrays)
    else:
        max_len = 0

    # Determine the new maximum length after adding the new array
    new_max_len = max(max_len, new_len)

    # Pad all existing arrays to match the new maximum length
    for i in range(len(arrays)):
        if len(arrays[i]) < new_max_len:
            pad_width = new_max_len - len(arrays[i])
            arrays[i] = np.pad(arrays[i], (0, pad_width), constant_values=np.nan)

    # Pad the new array if it’s shorter
    if new_len < new_max_len:
        new_array = np.pad(new_array, (0, new_max_len - new_len), constant_values=np.nan)

    # Append the new array
    arrays.append(new_array)

    return arrays