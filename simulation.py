import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from math import sqrt, atan2, pi, floor, cos, sin
import csv

import numba as nb
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
def ray_march_front(city_core, cx, cy, num_angles, ray_step=0.5):
    """
    Ray marching method: March along each angle from center and find furthest occupied cell.
    GUARANTEES all num_angles bins are occupied (at minimum, finds the center).
    
    This is the method used in the Jupyter notebook Eden growth code.
    """
    grid_size = city_core.shape[0]
    two_pi = 2.0 * np.pi
    bin_width = two_pi / num_angles
    
    r_max = np.zeros(num_angles, dtype=float64)
    max_radius = min(cx, cy, grid_size - 1 - cx, grid_size - 1 - cy)
    
    for k in range(num_angles):
        theta = (k + 0.5) * bin_width  # Center of each bin
        ct = cos(theta)
        st = sin(theta)
        
        r = 0.0
        r_last = 0.0
        
        # March outward along this ray
        while r <= max_radius:
            x = cx + r * ct
            y = cy + r * st
            
            # Round to nearest grid cell
            ix = int(round(x))
            iy = int(round(y))
            
            # Bounds check
            if ix < 0 or ix >= grid_size or iy < 0 or iy >= grid_size:
                break
            
            # If occupied, record this radius
            if city_core[iy, ix] == 1:  # Note: city_core[row, col]
                r_last = r
            
            r += ray_step
        
        r_max[k] = r_last
    
    return r_max


@nb.njit(fastmath=True, cache=True)
def extract_front_points_ray_march(r_max, num_angles):
    """
    Extract front points from ray-marched r_max.
    Unlike the boundary binning version, ALL bins should be occupied.
    
    Returns r and theta arrays that are already sorted by theta.
    """
    two_pi = 2.0 * np.pi
    bin_width = two_pi / num_angles
    
    # With ray marching, all bins should have r > 0, but check anyway
    valid_count = 0
    for i in range(num_angles):
        if r_max[i] > 0:
            valid_count += 1
    
    if valid_count == 0:
        # Return empty arrays if somehow no valid bins
        return np.zeros(0, dtype=float64), np.zeros(0, dtype=float64)
    
    r_front = np.zeros(valid_count, dtype=float64)
    theta_front = np.zeros(valid_count, dtype=float64)
    
    idx = 0
    for i in range(num_angles):
        if r_max[i] > 0:
            r_front[idx] = r_max[i]
            theta_front[idx] = (i + 0.5) * bin_width
            idx += 1
    
    return r_front, theta_front


# Keep the old boundary binning methods for comparison
@nb.njit(fastmath=True, cache=True)
def initialize_front_from_core(core_ys, core_xs, core_count, cx, cy, num_angles):
    """Build initial r_max from existing core (one-time computation)."""
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
def update_front_incremental(r_max, core_ys, core_xs, start_idx, end_idx, cx, cy, num_angles):
    """FAST INCREMENTAL UPDATE: Only update r_max for newly grown cells."""
    two_pi = 2 * np.pi
    bin_width = two_pi / num_angles
    
    for i in range(start_idx, end_idx):
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
    """
    Extract valid front points from r_max array.
    IMPORTANT: Output is ALREADY SORTED by theta (no additional sorting needed!)
    """
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


@nb.njit(fastmath=True, cache=True)
def extract_raw_boundary(city_core, cx, cy):
    """
    Extract front without binning - use all boundary cells directly.
    Returns points sorted by angle.
    """
    grid_size = city_core.shape[0]
    two_pi = 2 * np.pi
    
    # First pass: count boundary cells
    boundary_count = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if city_core[y, x] == 1:
                is_boundary = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid_size and 0 <= nx < grid_size:
                            if city_core[ny, nx] == 0:
                                is_boundary = True
                                break
                        else:
                            is_boundary = True
                            break
                    if is_boundary:
                        break
                
                if is_boundary:
                    boundary_count += 1
    
    if boundary_count == 0:
        # Return explicitly typed empty arrays (FIX for Numba)
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    
    # Second pass: collect boundary cells
    r_front = np.zeros(boundary_count, dtype=np.float64)
    theta_front = np.zeros(boundary_count, dtype=np.float64)
    
    idx = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if city_core[y, x] == 1:
                is_boundary = False
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid_size and 0 <= nx < grid_size:
                            if city_core[ny, nx] == 0:
                                is_boundary = True
                                break
                        else:
                            is_boundary = True
                            break
                    if is_boundary:
                        break
                
                if is_boundary:
                    xx, yy = float64(x), float64(y)
                    dx_val = xx - cx
                    dy_val = yy - cy
                    r_front[idx] = sqrt(dx_val * dx_val + dy_val * dy_val)
                    theta_front[idx] = (atan2(dy_val, dx_val) + two_pi) % two_pi
                    idx += 1
    
    # Sort by theta
    sort_idx = np.argsort(theta_front)
    
    return r_front[sort_idx], theta_front[sort_idx]


@nb.njit(fastmath=True, cache=True)
def grow_batch_optimized(urban, city_core, boundary_y, boundary_x, boundary_count,
                         core_ys, core_xs, core_count, batch_size, grid_size, cx, cy):
    """Optimized batch growth with deferred front updates."""
    occupied = np.zeros(grid_size * grid_size, dtype=np.int8)
    
    for i in range(boundary_count):
        idx = boundary_y[i] * grid_size + boundary_x[i]
        occupied[idx] = 1
    
    cells_grown = 0
    
    for _ in range(batch_size):
        if boundary_count == 0:
            break
            
        pick_idx = np.random.randint(boundary_count)
        r, c = boundary_y[pick_idx], boundary_x[pick_idx]
        
        urban[r, c] = 1
        city_core[r, c] = 1
        
        core_ys[core_count] = r
        core_xs[core_count] = c
        core_count += 1
        cells_grown += 1
        
        boundary_y[pick_idx] = boundary_y[boundary_count - 1]
        boundary_x[pick_idx] = boundary_x[boundary_count - 1]
        boundary_count -= 1
        
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if urban[nr, nc] == 0:
                        idx = nr * grid_size + nc
                        if occupied[idx] == 0:
                            boundary_y[boundary_count] = nr
                            boundary_x[boundary_count] = nc
                            boundary_count += 1
                            occupied[idx] = 1
    
    return boundary_count, core_count, cells_grown


def simulate(
    grid_size=401,
    timesteps=10000,
    k=10,
    output_file=None,
    sampling=0.25,
    batch_size_mode='single',
    batch_size_param=1,
    metric_interval=None,
    metric_timesteps=None,
    visualize_interval=100,
    adaptive_binning=False,
    use_binning=True,
    use_ray_marching=False,  # NEW PARAMETER
    ray_step=0.5  # NEW PARAMETER
):
    """
    Enhanced Eden growth simulation with optional ray marching for front extraction.
    
    NEW PARAMETERS:
    - use_ray_marching (bool): If True, use ray marching method to ensure all bins occupied.
                               If False, use boundary binning method (original).
    - ray_step (float): Step size for ray marching (default 0.5 pixels).
    
    Ray marching guarantees all angular bins are occupied, eliminating zero-inflation
    in variance calculations. However, it's slower than boundary binning.
    """
    
    assert grid_size % 2 == 1, "grid_size must be odd."
    
    np.random.seed(42)
    urban = np.zeros((grid_size, grid_size), dtype=np.uint8)
    city_core = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    cx = cy = grid_size // 2
    urban[cy, cx] = 1
    city_core[cy, cx] = 1
    
    max_boundary = grid_size * grid_size
    max_core = grid_size * grid_size
    
    boundary_y = np.zeros(max_boundary, dtype=np.int32)
    boundary_x = np.zeros(max_boundary, dtype=np.int32)
    core_ys = np.zeros(max_core, dtype=np.int32)
    core_xs = np.zeros(max_core, dtype=np.int32)
    
    core_ys[0] = cy
    core_xs[0] = cx
    core_count = 1
    
    boundary_count = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            nr = cy + dr
            nc = cx + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                boundary_y[boundary_count] = nr
                boundary_x[boundary_count] = nc
                boundary_count += 1
    
    total_cells = grid_size * grid_size
    two_pi = 2 * np.pi
    
    # Output setup
    if output_file:
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['time', 'N', 'w', 'mean_radius', 'urban_fraction'])
    else:
        writer = None
        w_results = []
        mean_results = []
        N_sectors_results = []
        urban_fraction_list = []
    
    # Binning configuration
    min_bins = 20
    max_angles = 50000
    
    if not use_binning:
        adaptive_binning = False
        print("Binning DISABLED: Using raw boundary cells")
    elif not adaptive_binning:
        num_angles_fixed = 10000
        print(f"Fixed binning mode: {num_angles_fixed} bins")
    
    # Metric sampling configuration
    if metric_timesteps is not None:
        metric_set = set(int(t) for t in metric_timesteps)
        compute_metrics_every_batch = False
        use_custom_timesteps = True
        print(f"Custom metric sampling: {len(metric_timesteps)} timesteps")
    elif metric_interval is None:
        compute_metrics_every_batch = True
        use_custom_timesteps = False
        metric_set = None
        metric_interval = 1
        print(f"Metrics: computed after every batch")
    else:
        compute_metrics_every_batch = False
        use_custom_timesteps = False
        metric_set = None
        print(f"Sparse metrics: computed every {metric_interval} timesteps")
    
    last_metric_t = -1 if not use_custom_timesteps else None
    
    # Batch size configuration
    if batch_size_mode == 'single' or batch_size_mode == 0 or batch_size_mode == 1:
        use_single_cell = True
        use_adaptive_batch = False
        fixed_batch_size = 1
        print(f"Batch mode: SINGLE CELL")
    elif batch_size_mode == 'adaptive':
        use_single_cell = False
        use_adaptive_batch = True
        batch_proportion = float(batch_size_param)
        print(f"Batch mode: ADAPTIVE ({batch_proportion*100:.1f}% of perimeter)")
    elif batch_size_mode == 'fixed' or isinstance(batch_size_mode, int):
        use_single_cell = False
        use_adaptive_batch = False
        if isinstance(batch_size_mode, int):
            fixed_batch_size = batch_size_mode
        else:
            fixed_batch_size = int(batch_size_param)
        print(f"Batch mode: FIXED ({fixed_batch_size} cells per batch)")
    
    # Front extraction method
    if use_ray_marching:
        print(f"Front extraction: RAY MARCHING (step={ray_step}) - ALL bins guaranteed occupied")
    else:
        print(f"Front extraction: BOUNDARY BINNING (sparse bins possible)")
    
    # Initialize front (only for boundary binning method)
    if use_binning and not use_ray_marching:
        if adaptive_binning:
            est_radius = max(10.0, grid_size / 20)
            num_angles = int(sampling * two_pi * est_radius)
            num_angles = min(max(num_angles, min_bins), max_angles)
        else:
            num_angles = num_angles_fixed
        
        r_max = initialize_front_from_core(core_ys, core_xs, core_count, cx, cy, num_angles)
        print(f"Initial front: {num_angles} angular bins (binning enabled)")
    else:
        r_max = None
        num_angles = num_angles_fixed if not adaptive_binning else 0
        if use_binning and use_ray_marching:
            if not adaptive_binning:
                print(f"Ray marching with {num_angles} fixed bins")
            else:
                print(f"Ray marching with adaptive binning")
    
    batch_size = fixed_batch_size if not use_adaptive_batch else 10
    t = 0
    batch_step = 0
    urban_cells = np.count_nonzero(urban)
    
    print(f"Starting simulation: {grid_size}x{grid_size} grid, {timesteps} timesteps")
    print(f"Initial boundary: {boundary_count} cells (buffer: {max_boundary}), Initial core: {core_count} cells")
    
    # Main loop
    while t < timesteps:
        batch_step += 1
        
        if use_single_cell:
            current_batch = 1
        elif use_adaptive_batch:
            current_batch = batch_size
        else:
            current_batch = min(fixed_batch_size, timesteps - t)
        
        current_batch = min(current_batch, timesteps - t)
        core_count_before = core_count
        
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
        
        # Determine if metrics should be computed
        if use_custom_timesteps:
            should_compute_metrics = (t in metric_set)
        elif compute_metrics_every_batch:
            should_compute_metrics = True
        else:
            should_compute_metrics = (t - last_metric_t >= metric_interval)
        
        # Compute metrics
        if core_count > 0 and should_compute_metrics:
            if use_binning:
                # === RAY MARCHING METHOD (NEW) ===
                if use_ray_marching:
                    # Update num_angles if adaptive
                    if adaptive_binning:
                        recent_idx = max(0, core_count - 100)
                        sample_xs = core_xs[recent_idx:core_count]
                        sample_ys = core_ys[recent_idx:core_count]
                        sample_dists = np.sqrt((sample_xs - cx)**2 + (sample_ys - cy)**2)
                        est_radius = np.max(sample_dists) if len(sample_dists) > 0 else 10.0
                        
                        num_angles = int(sampling * two_pi * est_radius)
                        num_angles = min(max(num_angles, min_bins), max_angles)
                    # else: num_angles already set to num_angles_fixed
                    
                    # Ray march to get front (ALL bins will be occupied)
                    r_max = ray_march_front(city_core, float(cx), float(cy), num_angles, ray_step)
                    r_front, theta_front = extract_front_points_ray_march(r_max, num_angles)
                
                # === BOUNDARY BINNING METHOD (ORIGINAL) ===
                else:
                    # Use angular binning with boundary detection
                    if adaptive_binning:
                        recent_idx = max(0, core_count - 100)
                        sample_xs = core_xs[recent_idx:core_count]
                        sample_ys = core_ys[recent_idx:core_count]
                        sample_dists = np.sqrt((sample_xs - cx)**2 + (sample_ys - cy)**2)
                        est_radius = np.max(sample_dists) if len(sample_dists) > 0 else 10.0
                        
                        new_num_angles = int(sampling * two_pi * est_radius)
                        new_num_angles = min(max(new_num_angles, min_bins), max_angles)
                        
                        if new_num_angles != num_angles:
                            num_angles = new_num_angles
                            r_max = initialize_front_from_core(core_ys, core_xs, core_count, cx, cy, num_angles)
                        else:
                            r_max = update_front_incremental(r_max, core_ys, core_xs,
                                                             core_count_before, core_count,
                                                             cx, cy, num_angles)
                    else:
                        # FIXED: Don't reassign num_angles here!
                        r_max = update_front_incremental(r_max, core_ys, core_xs,
                                                         core_count_before, core_count,
                                                         cx, cy, num_angles)
                    
                    r_front, theta_front = extract_front_points(r_max, num_angles)
            else:
                # No binning: extract raw boundary
                r_front, theta_front = extract_raw_boundary(city_core, cx, cy)
            
            # Update batch size if adaptive
            if use_adaptive_batch and len(r_front) > 0:
                mean_r = np.mean(r_front)
                batch_size = int(batch_proportion * mean_r * two_pi)
                batch_size = max(1, min(batch_size, 10000))
            
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
                theta_sorted = theta_front  # Already sorted
                r_sorted = r_front
                
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
            
            if not use_custom_timesteps:
                last_metric_t = t
        
        # Visualization
        if visualize_interval and batch_step % visualize_interval == 0:
            urban_frac = urban_cells / total_cells
            batch_info = f"batch={current_batch}" if not use_single_cell else "single-cell"
            
            if use_binning:
                if use_ray_marching:
                    bins_info = f"ray_march: bins={num_angles if core_count > 0 else 0} (all occupied)"
                else:
                    bins_info = f"boundary: bins={num_angles if core_count > 0 else 0}"
            else:
                bins_info = "raw boundary (no binning)"
            
            print(f"Batch {batch_step} | t={t}/{timesteps} | "
                  f"filled {urban_frac*100:.2f}% | "
                  f"{batch_info} | "
                  f"boundary={boundary_count} | "
                  f"{bins_info}")
    
    # Close CSV
    if writer is not None:
        f.close()
    
    # Output
    if writer is None:
        results = {
            "N_sectors": N_sectors_results,
            "w_t": w_results,
            "mean_radius": mean_results,
            "urban_fraction": urban_fraction_list
        }
    else:
        results = {"output_file": output_file}
    
    print(f"✓ Simulation complete!")
    
    return results