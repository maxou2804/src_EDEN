import numpy as np
from scipy.ndimage import binary_dilation
from math import sqrt
from scipy.sparse import issparse
import matplotlib.pyplot as plt


def simulate(
    grid_size,
    urban,
    city_core,
    timesteps,
    N_sectors_list,
    prob=1.0,
    nu=0.0,
    num_angles=1000,
):
    """
    Fast dense urban growth simulation using extended Eden model.
    Adds exactly one point per timestep to both urban and city_core.
    Computes w(t), mean radius(t), and urban fraction(t) for each N_sectors.
    Front is defined by maximum radial distance of city_core cells at uniform angular bins.
    Visualizes front and city_core boundary points for comparison.

    Parameters
    ----------
    grid_size : int
        Size of the square grid.
    urban : np.ndarray (uint8)
        Binary grid of existing urban cells (1 = urban, 0 = empty).
    city_core : np.ndarray (uint8)
        Binary grid of the city core (seed area).
    timesteps : int
        Number of simulation steps.
    N_sectors_list : list[int]
        List of sector counts to compute w for.
    prob : float
        Growth probability (default = 1.0). Currently unused.
    nu : float
        Exponent for extended Eden model probabilities (p ∝ (n_neighbors/4)^nu).
        nu=0.0 recovers uniform selection (standard Eden A model).
        For isotropy on square lattice, use nu ≈ 1.72.
    num_angles : int
        Number of angular bins for front definition (default = 1000).

    Returns
    -------
    results : dict
        {
            "N_sectors": list of N values,
            "w_t": list of arrays (one per N),
            "mean_radius": list of arrays (one per N),
            "urban_fraction": array (same for all N)
        }
    """

    # ---------- INPUT VALIDATION ----------
    if not isinstance(urban, np.ndarray) or not isinstance(city_core, np.ndarray):
        raise TypeError("urban and city_core must be NumPy arrays")

    if urban.shape != city_core.shape or urban.shape[0] != urban.shape[1]:
        raise ValueError("urban and city_core must be square arrays of the same size")

    # Ensure correct dtype and binary format
    urban = (urban > 0).astype(np.uint8)
    city_core = (city_core > 0).astype(np.uint8)
    grid_size = urban.shape[0]

    # Ensure city_core ⊆ urban
    if not np.all(city_core <= urban):
        raise ValueError("city_core must be inside (subset of) urban")

    # ---------- INITIALIZATION ----------
    total_cells = grid_size * grid_size
    urban_fraction = {N: np.zeros(timesteps) for N in N_sectors_list}
    w_results = {N: np.zeros(timesteps) for N in N_sectors_list}
    mean_results = {N: np.zeros(timesteps) for N in N_sectors_list}

    # Initial boundary (cells adjacent to urban)
    boundary = binary_dilation(urban) & (urban == 0)
    boundary_coords = np.argwhere(boundary)

    # Fixed center at grid midpoint
    cy, cx = grid_size / 2, grid_size / 2

    # Precompute angular bins for front sampling
    theta_bins = np.linspace(0, 2 * np.pi, num_angles + 1)  # Include endpoint
    bin_width = 2 * np.pi / num_angles

    # Precompute angular masks for each N (for sector-based roughness)
    Y_full, X_full = np.ogrid[:grid_size, :grid_size]
    dx = X_full - cx
    dy = Y_full - cy
    theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    sector_masks_dict = {}
    for N in N_sectors_list:
        masks = []
        sector_angle = 360 / N
        for i in range(N):
            tmin = i * sector_angle
            tmax = (i + 1) * sector_angle
            masks.append((theta >= tmin) & (theta < tmax))
        sector_masks_dict[N] = masks

    # ---------- MAIN SIMULATION LOOP ----------
    for t in range(timesteps):
        if len(boundary_coords) == 0:
            print("No more boundary cells to grow.")
            break

        # Compute number of urban neighbors (n) for each boundary cell
        rows = boundary_coords[:, 0]
        cols = boundary_coords[:, 1]
        n = np.zeros(len(rows), dtype=int)

        # Up
        mask = rows > 0
        n[mask] += urban[rows[mask] - 1, cols[mask]]

        # Down
        mask = rows < grid_size - 1
        n[mask] += urban[rows[mask] + 1, cols[mask]]

        # Left
        mask = cols > 0
        n[mask] += urban[rows[mask], cols[mask] - 1]

        # Right
        mask = cols < grid_size - 1
        n[mask] += urban[rows[mask], cols[mask] + 1]

        # Select cell using extended Eden probabilities
        if nu == 0.0:
            idx = np.random.randint(len(boundary_coords))
        else:
            weights = (n / 4.0) ** nu
            weight_sum = np.sum(weights)
            if weight_sum == 0:
                idx = np.random.randint(len(boundary_coords))
            else:
                probs = weights / weight_sum
                idx = np.random.choice(len(boundary_coords), p=probs)

        r, c = boundary_coords[idx]
        urban[r, c] = 1
        city_core[r, c] = 1  # Directly add the selected point to city_core

        # Update boundary incrementally
        local = np.zeros_like(urban)
        local[r, c] = 1
        new_boundary = binary_dilation(local) & (urban == 0)
        boundary = (boundary | new_boundary) & (urban == 0)
        boundary_coords = np.argwhere(boundary)

        # Update front: Find maximum radial distance using city_core cells
        core_coords = np.argwhere(city_core)
        if len(core_coords) > 0:
            ys_core, xs_core = core_coords[:, 0], core_coords[:, 1]
            dist_core = np.sqrt((xs_core - cx) ** 2 + (ys_core - cy) ** 2)
            theta_core = np.arctan2(ys_core - cy, xs_core - cx)
            # Normalize angles to [0, 2π), handling numerical precision
            theta_core = (theta_core + 2 * np.pi) % (2 * np.pi)
            
            # Assign city_core cells to angular bins using np.digitize
            bin_indices = np.digitize(theta_core, theta_bins, right=False) - 1
            bin_indices = np.clip(bin_indices, 0, num_angles - 1)
            
            # Compute maximum distance per bin
            r_max = np.zeros(num_angles)
            for i in range(num_angles):
                mask = bin_indices == i
                if np.any(mask):
                    r_max[i] = dist_core[mask].max()
            
            # Filter valid front points
            valid = r_max > 0
            r_front = r_max[valid]
            theta_front = theta_bins[valid] + bin_width / 2

            # Compute city_core boundary for comparison
            front_mask = binary_dilation(city_core) & (city_core == 0)
            ys_boundary, xs_boundary = np.nonzero(front_mask)
            if len(xs_boundary) > 0:
                r_boundary = np.sqrt((xs_boundary - cx) ** 2 + (ys_boundary - cy) ** 2)
                theta_boundary = (np.arctan2(ys_boundary - cy, xs_boundary - cx) + 2 * np.pi) % (2 * np.pi)
            else:
                r_boundary = np.array([])
                theta_boundary = np.array([])
        else:
            r_front = np.array([])
            theta_front = np.array([])
            r_boundary = np.array([])
            theta_boundary = np.array([])

        # Metrics
        if len(r_front) > 0:
            for N in N_sectors_list:
                sector_ids = (theta_front / (2 * np.pi / N)).astype(int)
                stds, means = [], []
                urban_fraction[N][t] = np.count_nonzero(urban) / total_cells

                for i in range(N):
                    s_dist = r_front[sector_ids == i]
                    if len(s_dist) > 0:
                        mean_r = s_dist.mean()
                        std_r = np.mean((s_dist - mean_r) ** 2)
                    else:
                        mean_r, std_r = 0, 0
                    means.append(mean_r)
                    stds.append(std_r)
                w_results[N][t] = np.mean(stds) ** 0.5
                mean_results[N][t] = np.mean(means)
        else:
            for N in N_sectors_list:
                w_results[N][t] = 0
                mean_results[N][t] = 0

        # Visualization and coordinate output
        if (t + 1) % 50 == 0 or t == timesteps - 1:
            print(f"timestep {t+1}/{timesteps} | filled {urban_fraction[20][t]*100:.2f}%")
            # Plot front and boundary points
            plt.figure(figsize=(5, 5))
            if len(r_front) > 0:
                plt.polar(theta_front, r_front, 'bo', markersize=2, label='Front (Max Radial)')
            if len(r_boundary) > 0:
                plt.polar(theta_boundary, r_boundary, 'rx', markersize=2, label='City_core Boundary')
            plt.title(f"Front vs. Boundary at t={t+1}")
            plt.legend()
            plt.show()

            # Print a subset of coordinates for comparison, focusing on cardinal angles
            print(f"Timestep {t+1}:")
            print(f"  Front points near cardinal angles (0, π/2, π, 3π/2):")
            cardinal_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            for target_theta in cardinal_angles:
                if len(r_front) > 0:
                    idx = np.argmin(np.abs((theta_front - target_theta + np.pi) % (2 * np.pi) - np.pi))
                    print(f"    theta≈{target_theta:.3f}, r={r_front[idx]:.3f}, actual theta={theta_front[idx]:.3f}")
            print(f"  Boundary points (first 5, polar coords):")
            for i in range(min(5, len(r_boundary))):
                print(f"    theta={theta_boundary[i]:.3f}, r={r_boundary[i]:.3f}")
            print(f"  Valid front bins: {len(r_front)}/{num_angles}")

    # ---------- OUTPUT ----------
    results = {
        "N_sectors": N_sectors_list,
        "w_t": [w_results[N] for N in N_sectors_list],
        "mean_radius": [mean_results[N] for N in N_sectors_list],
        "urban_fraction": [urban_fraction[N] for N in N_sectors_list]
    }

    return results