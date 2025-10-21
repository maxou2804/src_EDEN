import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from math import sqrt, atan2, pi, floor
import csv

import numba as nb
from numba import types
from numba.types import int64, float64

@nb.njit(fastmath=True, cache=True)
def compute_sector_stats(theta_sorted, r_sorted, N):
    """Numba-optimized sector statistics computation."""
    bin_edges = np.linspace(0, 2 * np.pi, N + 1)
    means = np.zeros(N, dtype=float64)
    vars_ = np.zeros(N, dtype=float64)
    j = 0
    for i in range(N):
        sum_r = 0.0
        sum_sq = 0.0
        count = 0
        while j < len(theta_sorted) and theta_sorted[j] < bin_edges[i + 1]:
            r = r_sorted[j]
            sum_r += r
            sum_sq += r * r
            count += 1
            j += 1
        if count > 0:
            mean = sum_r / count
            means[i] = mean
            if count > 1:
                vars_[i] = (sum_sq - (sum_r ** 2 / count)) / (count - 1)
            else:
                vars_[i] = 0.0
        else:
            means[i] = 0.0
            vars_[i] = 0.0
    return means, vars_


@nb.njit(fastmath=True, cache=True)
def grow_batch_optimized(
    urban, city_core, 
    boundary_y, boundary_x, boundary_count,
    core_ys, core_xs, core_count,
    batch_size, grid_size, cx, cy
):
    """
    Optimized batch growth with deferred front updates.
    Returns: new_boundary_count, new_core_count, cells_grown
    """
    occupied = np.zeros(grid_size * grid_size, dtype=np.int8)
    
    # Mark initial boundary
    for i in range(boundary_count):
        idx = boundary_y[i] * grid_size + boundary_x[i]
        occupied[idx] = 1
    
    cells_grown = 0
    
    for _ in range(batch_size):
        if boundary_count == 0:
            break
            
        # Pick random boundary cell
        pick_idx = np.random.randint(boundary_count)
        r, c = boundary_y[pick_idx], boundary_x[pick_idx]
        
        # Grow cell
        urban[r, c] = 1
        city_core[r, c] = 1
        
        # Store in core arrays
        core_ys[core_count] = r
        core_xs[core_count] = c
        core_count += 1
        cells_grown += 1
        
        # Swap-remove from boundary
        boundary_y[pick_idx] = boundary_y[boundary_count - 1]
        boundary_x[pick_idx] = boundary_x[boundary_count - 1]
        boundary_count -= 1
        
        # Add neighbors
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                if abs(dr) + abs(dc) == 2:  # Skip diagonals
                    continue
                    
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size and city_core[nr, nc] == 0:
                    nidx = nr * grid_size + nc
                    if occupied[nidx] == 0:
                        boundary_y[boundary_count] = nr
                        boundary_x[boundary_count] = nc
                        occupied[nidx] = 1
                        boundary_count += 1
    
    return boundary_count, core_count, cells_grown


@nb.njit(fastmath=True, cache=True)
def compute_front_from_core(core_ys, core_xs, core_count, cx, cy, num_angles):
    """
    LAZY METRICS: Compute front from stored core coordinates.
    Returns r_max array and valid mask.
    """
    two_pi = 2 * np.pi
    bin_width = two_pi / num_angles
    r_max = np.zeros(num_angles, dtype=float64)
    
    for i in range(core_count):
        x, y = float64(core_xs[i]), float64(core_ys[i])
        dx = x - cx
        dy = y - cy
        dist = sqrt(dx * dx + dy * dy)
        theta = (atan2(dy, dx) + two_pi) % two_pi
        bin_idx = int64(floor(theta / bin_width))
        if bin_idx >= num_angles:
            bin_idx = num_angles - 1
        if dist > r_max[bin_idx]:
            r_max[bin_idx] = dist
    
    return r_max


@nb.njit(fastmath=True, cache=True)
def extract_front_points(r_max, num_angles):
    """Extract valid front points from r_max array."""
    two_pi = 2 * np.pi
    bin_width = two_pi / num_angles
    
    # Count valid bins
    valid_count = 0
    for i in range(num_angles):
        if r_max[i] > 0:
            valid_count += 1
    
    r_front = np.zeros(valid_count, dtype=float64)
    theta_front = np.zeros(valid_count, dtype=float64)
    
    idx = 0
    for i in range(num_angles):
        if r_max[i] > 0:
            r_front[idx] = r_max[i]
            theta_front[idx] = (i + 0.5) * bin_width
            idx += 1
    
    return r_front, theta_front


def simulate(
    grid_size,
    urban,
    city_core,
    timesteps,
    k,
    prob=1.0,
    sampling=1,
    max_bins=20000,
    min_bins=100,
    adaptive_binning=True,
    batch_size_mode='adaptive',
    batch_size_param=0.1,
    output_file=None,
    visualize_interval=1000
):
    """
    Optimized Eden growth simulation with:
    - Numba-optimized growth loop
    - Lazy metric computation (deferred front updates)
    - Optimized boundary buffer with pre-allocated arrays
    - Configurable adaptive binning strategy
    
    Parameters
    ----------
    grid_size : int
        Size of the square grid.
    urban : np.ndarray (uint8)
        Binary grid of existing urban cells.
    city_core : np.ndarray (uint8)
        Binary grid of the city core.
    timesteps : int
        Number of simulation steps.
    k : int
        Number of N_sectors values per timestep.
    prob : float
        Growth probability (unused, kept for compatibility).
    sampling : float
        Sampling factor for angular bins: num_bins ≈ sampling * 2π * radius.
        For roughness measurements: Use sampling ≥ 1.0 to ensure proper resolution.
        Memory-limited cases: 0.5-0.8 minimum, but verify bin_width < 2 cells.
        Recommended: 1.0-1.5 (proper sampling), 0.5-0.8 (memory-limited).
    max_bins : int, optional
        Maximum number of angular bins. If None, uses 10000 for grids > 512,
        otherwise sampling * 2π * grid_size. Prevents memory issues on large grids.
    min_bins : int
        Minimum number of angular bins (default: 100).
    adaptive_binning : bool
        If True, recompute num_bins each batch based on current radius.
        If False, use fixed binning based on initial estimate.
    output_file : str, optional
        CSV file path for streaming output.
    visualize_interval : int
        Print progress every N batches.
    
    Returns
    -------
    results : dict
        Simulation results or output file path.
    """
    
    # ---------- INPUT VALIDATION ----------
    if not isinstance(urban, np.ndarray) or not isinstance(city_core, np.ndarray):
        raise TypeError("urban and city_core must be NumPy arrays")
    
    if urban.shape != city_core.shape or urban.shape[0] != urban.shape[1]:
        raise ValueError("urban and city_core must be square arrays of the same size")
    
    urban = urban.copy().astype(np.uint8)
    city_core = city_core.copy().astype(np.uint8)
    grid_size = urban.shape[0]
    
    if not np.all(city_core <= urban):
        raise ValueError("city_core must be inside urban")
    
    # ---------- INITIALIZATION ----------
    total_cells = grid_size * grid_size
    cy, cx = grid_size / 2.0, grid_size / 2.0
    two_pi = 2 * np.pi
    
    # CSV streaming
    writer = None
    if output_file is not None:
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['time', 'N', 'w', 'mean_radius', 'urban_fraction'])
    
    if writer is None:
        w_results = []
        mean_results = []
        N_sectors_results = []
        urban_fraction_list = []
    
    # OPTIMIZED BOUNDARY BUFFER: Pre-allocated arrays
    max_boundary = min(grid_size * 4, 100000)  # Cap for memory
    boundary_y = np.zeros(max_boundary, dtype=np.int64)
    boundary_x = np.zeros(max_boundary, dtype=np.int64)
    
    # Initialize boundary
    boundary_mask = binary_dilation(urban) & (urban == 0)
    initial_boundary = np.argwhere(boundary_mask)
    boundary_count = min(len(initial_boundary), max_boundary)
    boundary_y[:boundary_count] = initial_boundary[:boundary_count, 0]
    boundary_x[:boundary_count] = initial_boundary[:boundary_count, 1]
    
    # LAZY METRICS: Store core coordinates
    initial_core = np.argwhere(city_core)
    C0 = len(initial_core)
    max_core = C0 + timesteps
    core_ys = np.zeros(max_core, dtype=np.int64)
    core_xs = np.zeros(max_core, dtype=np.int64)
    
    if C0 > 0:
        core_ys[:C0] = initial_core[:, 0]
        core_xs[:C0] = initial_core[:, 1]
    core_count = C0
    
    # ADAPTIVE BINNING: Scale with radius but cap at max_angles
    max_angles = 10000  # Maximum bins to prevent excessive memory usage
    
    # Determine max_angles based on grid size and user input
    if max_bins is None:
        # Auto-select: cap for large grids to prevent memory issues
        if grid_size > 512:
            max_angles = 10000
        else:
            # For smaller grids, allow full resolution
            max_angles = int(sampling * two_pi * grid_size)
    else:
        max_angles = max_bins
    
    # Ensure min_bins is respected
    max_angles = max(max_angles, min_bins)
    
    # For fixed binning mode, compute once
    if not adaptive_binning:
        # Estimate final radius (diagonal / 2)
        est_final_radius = grid_size / np.sqrt(2)
        num_angles_fixed = int(sampling * two_pi * est_final_radius)
        num_angles_fixed = min(max(num_angles_fixed, min_bins), max_angles)
        print(f"Fixed binning mode: {num_angles_fixed} bins")
    
    # ---------- MAIN SIMULATION LOOP ----------
    batch_size = 10  # Initial batch size (will be adaptive)
    t = 0
    batch_step = 0
    urban_cells = np.count_nonzero(urban)
    
    print(f"Starting simulation: {grid_size}x{grid_size} grid, {timesteps} timesteps")
    print(f"Initial boundary: {boundary_count} cells, Initial core: {core_count} cells")
    print(f"Binning: sampling={sampling}, max_bins={max_angles}, min_bins={min_bins}, adaptive={adaptive_binning}")
    
    while t < timesteps:
        batch_step += 1
        current_batch = min(batch_size, timesteps - t)
        
        # OPTIMIZED GROWTH (Numba-accelerated)
        boundary_count, core_count, cells_grown = grow_batch_optimized(
            urban, city_core,
            boundary_y, boundary_x, boundary_count,
            core_ys, core_xs, core_count,
            current_batch, grid_size, cx, cy
        )
        
        urban_cells += cells_grown
        t += cells_grown
        
        if t >= timesteps:
            t = timesteps
        
        # LAZY METRICS: Only compute when needed (after each batch)
        if core_count > 0:
            # Determine number of bins
            if adaptive_binning:
                # Estimate current radius for adaptive binning
                recent_idx = max(0, core_count - 100)
                sample_xs = core_xs[recent_idx:core_count]
                sample_ys = core_ys[recent_idx:core_count]
                sample_dists = np.sqrt((sample_xs - cx)**2 + (sample_ys - cy)**2)
                est_radius = np.max(sample_dists) if len(sample_dists) > 0 else 10.0
                
                # Scale with radius but respect bounds
                num_angles = int(sampling * two_pi * est_radius)
                num_angles = min(max(num_angles, min_bins), max_angles)
            else:
                # Use fixed binning
                num_angles = num_angles_fixed
            
            # DEFERRED FRONT UPDATE: Compute from stored coordinates
            r_max = compute_front_from_core(core_ys, core_xs, core_count, cx, cy, num_angles)
            r_front, theta_front = extract_front_points(r_max, num_angles)
            
            # Adapt batch size based on perimeter
            if len(r_front) > 0:
                mean_r = np.mean(r_front)
                batch_size = int(0.2 * mean_r * two_pi)
                batch_size = max(1, min(batch_size, 1000))  # Cap batch size
            
            # Calculate N_sectors list
            urban_frac = urban_cells / total_cells
            N_list = []
            if len(r_front) > 0:
                total_points = len(r_front)
                min_N = 4
                max_N = total_points // 5
                if max_N >= min_N:
                    num_N = min(k, max_N - min_N + 1)
                    N_list = np.logspace(np.log10(min_N), np.log10(max_N), num_N, dtype=int)
                    N_list = sorted(set(N_list.tolist()))
            
            # Compute metrics
            row_w = [0.0] * k
            row_mean = [0.0] * k
            row_N = [0] * k
            
            if len(r_front) > 0 and len(N_list) > 0:
                sort_idx = np.argsort(theta_front)
                theta_sorted = theta_front[sort_idx]
                r_sorted = r_front[sort_idx]
                
                for ii in range(min(len(N_list), k)):
                    N = N_list[ii]
                    means, vars_ = compute_sector_stats(theta_sorted, r_sorted, N)
                    row_w[ii] = np.mean(vars_) ** 0.5
                    row_mean[ii] = np.mean(means)
                    row_N[ii] = N
            
            # Write results
            if writer is not None:
                for ii in range(k):
                    if row_N[ii] > 0:
                        writer.writerow([t, row_N[ii], row_w[ii], row_mean[ii], urban_frac])
            else:
                w_results.append(row_w)
                mean_results.append(row_mean)
                N_sectors_results.append(row_N)
                urban_fraction_list.append(urban_frac)
        
        # Visualization
        if visualize_interval and batch_step % visualize_interval == 0:
            urban_frac = urban_cells / total_cells
            print(f"Batch {batch_step} | t={t}/{timesteps} | "
                  f"filled {urban_frac*100:.2f}% | "
                  f"batch_size={batch_size} | "
                  f"boundary={boundary_count} | "
                  f"bins={num_angles if core_count > 0 else 0}")
            
            # DIAGNOSTIC: Check sampling quality for roughness measurements
            if core_count > 0:
                mean_r = np.mean(r_front) if len(r_front) > 0 else 0
                if mean_r > 0:
                    arc_length = mean_r * two_pi / num_angles
                    quality = "EXCELLENT" if arc_length < 1.0 else \
                             "GOOD" if arc_length < 1.5 else \
                             "MARGINAL" if arc_length < 2.5 else \
                             "⚠️ UNDERSAMPLED"
                    print(f"        → Arc length: {arc_length:.2f} cells/bin ({quality})")
                    if arc_length > 2.0:
                        print(f"        → ⚠️  WARNING: Increase 'sampling' parameter for accurate roughness!")
    
    # Close CSV file
    if writer is not None:
        f.close()
    
    # ---------- OUTPUT ----------
    if writer is None:
        results = {
            "N_sectors": N_sectors_results,
            "w_t": w_results,
            "mean_radius": mean_results,
            "urban_fraction": urban_fraction_list
        }
    else:
        results = {"output_file": output_file}
    
    print(f"✓ Simulation complete! | Filled {urban_cells}/{total_cells} cells ({urban_cells/total_cells*100:.1f}%)")
    
    return results