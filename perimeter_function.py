import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple
from rasterio.transform import xy, Affine

sys.path.insert(0, str(Path(__file__).parent))
from wsf_evolution_lcc import BuiltAreaAnalyzer
from numba import jit, prange
from typing import List, Tuple


def extract_perimeter_from_bbox_optimized(data: np.ndarray,
                                          transform: Affine,
                                          n_sectors: int = 32,
                                          use_numba: bool = False) -> pd.DataFrame:
    """
    ULTRA-FAST perimeter extraction optimized for large N.
    
    Optimizations:
    - Pre-compute all LCC masks
    - Vectorized sector assignment
    - Single-pass processing
    - Memory-efficient operations
    
    Args:
        data: WSF Evolution array
        transform: Affine transform
        n_sectors: Number of sectors (works well even for N=10000!)
        use_numba: Use numba JIT compilation (even faster)
    
    Returns:
        DataFrame with perimeter points
    """
    print(f"\n🚀 ULTRA-FAST PERIMETER EXTRACTION")
    print(f"  Sectors: {n_sectors:,}")
    print(f"  Optimization: {'Numba JIT' if use_numba else 'Vectorized'}")
    print("-" * 70)
    
    analyzer = BuiltAreaAnalyzer()
    
    # ========================================================================
    # STEP 1: Calculate 1985 center (only once)
    # ========================================================================
    print("\nStep 1: Calculating 1985 center...")
    mask_1985 = analyzer.extract_year_mask(data, 1985)
    lcc_1985, size_1985 = analyzer.find_largest_connected_component(mask_1985)
    
    if size_1985 == 0:
        raise ValueError("No LCC found in 1985!")
    
    rows_1985, cols_1985 = np.where(lcc_1985 == 1)
    center_row = float(rows_1985.mean())
    center_col = float(cols_1985.mean())
    
    print(f"  Center: (row={center_row:.2f}, col={center_col:.2f})")
    
    # ========================================================================
    # STEP 2: Pre-compute all LCC masks (memory-efficient)
    # ========================================================================
    print("\nStep 2: Pre-computing LCC masks...")
    
    years = list(range(1985, 2016))
    year_masks = {}
    year_coords = {}
    
    for year in years:
        mask_year = analyzer.extract_year_mask(data, year)
        lcc_year, size_year = analyzer.find_largest_connected_component(mask_year)
        
        if size_year == 0:
            continue
        
        rows, cols = np.where(lcc_year == 1)
        year_masks[year] = (rows, cols)
        
        # Pre-compute angles and distances for this year
        dx = cols.astype(np.float64) - center_col
        dy = -(rows.astype(np.float64) - center_row)
        
        angles = np.arctan2(dy, dx)
        angles[angles < 0] += 2 * np.pi
        distances = np.sqrt(dx**2 + dy**2)
        
        year_coords[year] = (rows, cols, angles, distances)
        
        print(f"  {year}: {size_year:,} pixels", end='\r')
    
    print(f"\n  Pre-computed {len(year_coords)} years")
    
    # ========================================================================
    # STEP 3: Fast sector assignment and perimeter extraction
    # ========================================================================
    print(f"\nStep 3: Extracting perimeter for {n_sectors:,} sectors...")
    
    # Use the optimized extraction method
    if use_numba:
        try:
            import numba
            perimeter_data = _extract_with_numba(
                year_coords, n_sectors, center_row, center_col
            )
        except ImportError:
            print("  Warning: numba not available, using vectorized version")
            perimeter_data = _extract_vectorized(
                year_coords, n_sectors, center_row, center_col
            )
    else:
        perimeter_data = _extract_vectorized(
            year_coords, n_sectors, center_row, center_col
        )
    
    print(f"  Extracted {len(perimeter_data):,} perimeter points")
    
    # ========================================================================
    # STEP 4: Create DataFrame
    # ========================================================================
    print("\nStep 4: Creating DataFrame...")
    df = pd.DataFrame(perimeter_data)
    
    # Add geographic coordinates (vectorized)
    print("  Adding geographic coordinates...")
    lons, lats = xy(transform, df['row'].values, df['col'].values, offset='center')
    df['latitude'] = np.array(lats)
    df['longitude'] = np.array(lons)
    
    # Reorder columns
    df = df[[
        'year', 'sector', 'sector_angle_deg', 'sector_angle_rad',
        'row', 'col', 'latitude', 'longitude',
        'distance_pixels',
        'actual_angle_deg', 'actual_angle_rad',
        'center_row', 'center_col'
    ]]
    
    print(f"✓ Complete! {len(df):,} perimeter points")
    
    return df


def _extract_vectorized(year_coords: dict, 
                       n_sectors: int,
                       center_row: float,
                       center_col: float) -> list:
    """
    Vectorized perimeter extraction.
    
    Key optimization: Use np.digitize for O(n log k) sector assignment
    instead of O(n*k) loop-based assignment.
    """
    sector_width = 2 * np.pi / n_sectors
    
    # Pre-compute sector boundaries
    sector_boundaries = np.linspace(0, 2*np.pi, n_sectors + 1)
    
    all_perimeter_data = []
    
    for year, (rows, cols, angles, distances) in year_coords.items():
        # OPTIMIZATION: Use digitize for fast sector assignment
        # This is O(n log k) instead of O(n*k)!
        sector_indices = np.digitize(angles, sector_boundaries) - 1
        
        # Handle wrap-around (angles near 2π)
        sector_indices[sector_indices >= n_sectors] = 0
        
        # For each sector, find the point with maximum distance
        unique_sectors = np.unique(sector_indices)
        
        for sector_idx in unique_sectors:
            # Get points in this sector
            in_sector = sector_indices == sector_idx
            
            # Find max distance point (vectorized argmax)
            sector_distances = distances[in_sector]
            local_max_idx = sector_distances.argmax()
            
            # Get the actual indices
            sector_rows = rows[in_sector]
            sector_cols = cols[in_sector]
            sector_angles = angles[in_sector]
            
            max_row = sector_rows[local_max_idx]
            max_col = sector_cols[local_max_idx]
            max_distance = sector_distances[local_max_idx]
            max_angle = sector_angles[local_max_idx]
            
            # Calculate sector center angle
            sector_center_angle = (sector_idx + 0.5) * sector_width
            
            all_perimeter_data.append({
                'year': year,
                'sector': int(sector_idx),
                'sector_angle_rad': sector_center_angle,
                'sector_angle_deg': np.degrees(sector_center_angle),
                'row': int(max_row),
                'col': int(max_col),
                'distance_pixels': float(max_distance),
                'actual_angle_rad': float(max_angle),
                'actual_angle_deg': float(np.degrees(max_angle)),
                'center_row': center_row,
                'center_col': center_col
            })
    
    return all_perimeter_data


def _extract_with_numba(year_coords: dict,
                        n_sectors: int,
                        center_row: float,
                        center_col: float) -> list:
    """
    Numba-optimized extraction (fastest for very large N).
    
    Can be 2-3x faster than vectorized version for N > 1000.
    """
    try:
        from numba import jit
    except ImportError:
        print("Warning: numba not installed, falling back to vectorized version")
        return _extract_vectorized(year_coords, n_sectors, center_row, center_col)
    
    @jit(nopython=True)
    def find_max_per_sector(angles, distances, n_sectors):
        """JIT-compiled function to find max distance per sector"""
        sector_width = 2 * np.pi / n_sectors
        
        # Initialize arrays for max distance and index per sector
        max_distances = np.full(n_sectors, -1.0)
        max_indices = np.full(n_sectors, -1, dtype=np.int64)
        
        # Single pass through all points
        for i in range(len(angles)):
            angle = angles[i]
            dist = distances[i]
            
            # Determine sector
            sector = int(angle / sector_width)
            if sector >= n_sectors:
                sector = n_sectors - 1
            
            # Update max if needed
            if dist > max_distances[sector]:
                max_distances[sector] = dist
                max_indices[sector] = i
        
        return max_distances, max_indices
    
    all_perimeter_data = []
    sector_width = 2 * np.pi / n_sectors
    
    for year, (rows, cols, angles, distances) in year_coords.items():
        # Use JIT-compiled function
        max_distances, max_indices = find_max_per_sector(
            angles.astype(np.float64),
            distances.astype(np.float64),
            n_sectors
        )
        
        # Extract results
        for sector_idx in range(n_sectors):
            idx = max_indices[sector_idx]
            if idx >= 0:  # Sector has points
                sector_center_angle = (sector_idx + 0.5) * sector_width
                
                all_perimeter_data.append({
                    'year': year,
                    'sector': int(sector_idx),
                    'sector_angle_rad': sector_center_angle,
                    'sector_angle_deg': np.degrees(sector_center_angle),
                    'row': int(rows[idx]),
                    'col': int(cols[idx]),
                    'distance_pixels': float(max_distances[sector_idx]),
                    'actual_angle_rad': float(angles[idx]),
                    'actual_angle_deg': float(np.degrees(angles[idx])),
                    'center_row': center_row,
                    'center_col': center_col
                })
    
    return all_perimeter_data







import numpy as np
import pandas as pd
from numba import jit, prange
from typing import List, Tuple


def analyze_sectors_optimized(csv_filename: str, N_sector: List[int], years: np.ndarray) -> np.ndarray:
    """
    Analyze distance pixels by sector with optimized performance.
    
    For each year, calculate statistics for each N_sector configuration:
    - Group points by angular position: sector k contains points with angles in [k*2π/n, (k+1)*2π/n)
    - Calculate mean and std for each angular sector
    - Average these statistics across all sectors
    - Repeat for all N_sector values
    
    Parameters:
    -----------
    csv_filename : str
        Path to CSV file containing sector data
    N_sector : List[int]
        List of sector counts (different angular resolutions to analyze)
    years : np.ndarray
        Array of years to analyze
    
    Returns:
    --------
    np.ndarray
        Array of shape (2*len(years), len(N_sector)) where:
        - Row 2*i contains averaged standard deviations for year i
        - Row 2*i+1 contains averaged mean radii for year i
        - Each column corresponds to one N_sector configuration
    """
    # Read CSV file
    df = pd.read_csv(csv_filename)
    
    # Initialize output array: 2 rows per year (std, mean), one column per N_sector value
    output = np.full((2 * len(years), len(N_sector)), np.nan)
    
    # Process each year
    for year_idx, year in enumerate(years):
        # Filter data for current year
        year_data = df[df['year'] == year]
        
        if len(year_data) == 0:
            continue
        
        # Extract relevant columns as numpy arrays
        distances = year_data['distance_pixels'].values
        
        # Get actual angles - try different column names
        if 'actual_angle_rad' in year_data.columns:
            angles = year_data['actual_angle_rad'].values
        elif 'sector_angle_rad' in year_data.columns:
            angles = year_data['sector_angle_rad'].values
        else:
            # Calculate angles from row/col if not available
            if 'row' in year_data.columns and 'col' in year_data.columns:
                center_row = year_data['center_row'].iloc[0] if 'center_row' in year_data.columns else year_data['row'].mean()
                center_col = year_data['center_col'].iloc[0] if 'center_col' in year_data.columns else year_data['col'].mean()
                
                dy = year_data['row'].values - center_row
                dx = year_data['col'].values - center_col
                angles = np.arctan2(dy, dx)
                # Normalize to [0, 2π)
                angles = np.where(angles < 0, angles + 2*np.pi, angles)
            else:
                print(f"Warning: No angle information found for year {year}")
                continue
        
        # Process each N_sector configuration
        for n_idx, n_sectors in enumerate(N_sector):
            # Pre-calculate angular mask width
            angular_width = 2 * np.pi / n_sectors
            
            # Calculate statistics for this N_sector configuration using optimized function
            std_vals, mean_vals = compute_sector_stats_by_angle(
                angles, distances, n_sectors, angular_width
            )
            
            # Average over all sectors (ignoring NaN values)
            avg_std = np.nanmean(std_vals)
            avg_mean = np.nanmean(mean_vals)
            
            # Store averaged results
            output[2 * year_idx, n_idx] = avg_std
            output[2 * year_idx + 1, n_idx] = avg_mean
    
    return output


@jit(nopython=True, parallel=True)
def compute_sector_stats_by_angle(angles: np.ndarray, distances: np.ndarray, 
                                   n_sectors: int, angular_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for each angular sector using Numba optimization.
    
    Points are assigned to sectors based on their angular position:
    - Sector k contains points with angles in [k*angular_width, (k+1)*angular_width)
    
    Parameters:
    -----------
    angles : np.ndarray
        Array of angular positions in radians [0, 2π)
    distances : np.ndarray
        Array of distance values
    n_sectors : int
        Number of angular sectors
    angular_width : float
        Width of each sector in radians (2π / n_sectors)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (std_values, mean_values) for each angular sector
    """
    std_vals = np.zeros(n_sectors)
    mean_vals = np.zeros(n_sectors)
    
    # Parallel processing of sectors
    for sector_id in prange(n_sectors):
        # Define angular bounds for this sector
        angle_min = sector_id * angular_width
        angle_max = (sector_id + 1) * angular_width
        
        # Find points in this angular sector
        mask = (angles >= angle_min) & (angles < angle_max)
        sector_distances = distances[mask]
        
        if len(sector_distances) > 0:
            mean_vals[sector_id] = np.mean(sector_distances)
            std_vals[sector_id] = np.std(sector_distances)
        else:
            mean_vals[sector_id] = np.nan
            std_vals[sector_id] = np.nan
    
    return std_vals, mean_vals


@jit(nopython=True, parallel=True)
def compute_sector_stats(sectors: np.ndarray, distances: np.ndarray, n_sectors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for each sector using Numba optimization.
    
    Parameters:
    -----------
    sectors : np.ndarray
        Array of sector indices
    distances : np.ndarray
        Array of distance values
    n_sectors : int
        Number of sectors
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (std_values, mean_values) for each sector
    """
    std_vals = np.zeros(n_sectors)
    mean_vals = np.zeros(n_sectors)
    
    # Parallel processing of sectors
    for sector_id in prange(n_sectors):
        # Find indices for current sector
        mask = sectors == sector_id
        sector_distances = distances[mask]
        
        if len(sector_distances) > 0:
            mean_vals[sector_id] = np.mean(sector_distances)
            std_vals[sector_id] = np.std(sector_distances)
        else:
            mean_vals[sector_id] = np.nan
            std_vals[sector_id] = np.nan
    
    return std_vals, mean_vals


def analyze_sectors_vectorized(csv_filename: str, N_sector: List[int], years: np.ndarray) -> np.ndarray:
    """
    Fully vectorized version using pandas groupby with angular binning.
    
    For each year and each N_sector configuration:
    - Bin points by angular position: sector k = [k*2π/n, (k+1)*2π/n)
    - Calculate mean and std for each angular sector
    - Average these statistics across all sectors
    
    Parameters:
    -----------
    csv_filename : str
        Path to CSV file containing sector data
    N_sector : List[int]
        List of sector counts (different angular resolutions to analyze)
    years : np.ndarray
        Array of years to analyze
    
    Returns:
    --------
    np.ndarray
        Array of shape (2*len(years), len(N_sector)) where:
        - Row 2*i contains averaged standard deviations for year i
        - Row 2*i+1 contains averaged mean radii for year i
        - Each column corresponds to one N_sector configuration
    """
    # Read CSV file
    df = pd.read_csv(csv_filename)
    
    # Initialize output array
    output = np.full((2 * len(years), len(N_sector)), np.nan)
    
    # Process each year
    for year_idx, year in enumerate(years):
        # Filter data for current year
        year_data = df[df['year'] == year].copy()
        
        if len(year_data) == 0:
            continue
        
        # Get actual angles
        if 'actual_angle_rad' in year_data.columns:
            angles = year_data['actual_angle_rad'].values
        elif 'sector_angle_rad' in year_data.columns:
            angles = year_data['sector_angle_rad'].values
        else:
            # Calculate angles from row/col if not available
            if 'row' in year_data.columns and 'col' in year_data.columns:
                center_row = year_data['center_row'].iloc[0] if 'center_row' in year_data.columns else year_data['row'].mean()
                center_col = year_data['center_col'].iloc[0] if 'center_col' in year_data.columns else year_data['col'].mean()
                
                dy = year_data['row'].values - center_row
                dx = year_data['col'].values - center_col
                angles = np.arctan2(dy, dx)
                # Normalize to [0, 2π)
                angles = np.where(angles < 0, angles + 2*np.pi, angles)
            else:
                continue
        
        year_data['angle'] = angles
        
        # Process each N_sector configuration
        for n_idx, n_sectors in enumerate(N_sector):
            # Calculate angular width
            angular_width = 2 * np.pi / n_sectors
            
            # Assign each point to an angular sector
            year_data['angular_sector'] = (angles / angular_width).astype(int)
            
            # Handle edge case where angle = 2π maps to sector n_sectors (should be sector 0)
            year_data.loc[year_data['angular_sector'] >= n_sectors, 'angular_sector'] = n_sectors - 1
            
            # Group by angular sector and compute statistics
            grouped = year_data.groupby('angular_sector')['distance_pixels'].agg(['std', 'mean'])
            
            # Average over all sectors
            avg_std = grouped['std'].mean()
            avg_mean = grouped['mean'].mean()
            
            # Store results
            output[2 * year_idx, n_idx] = avg_std
            output[2 * year_idx + 1, n_idx] = avg_mean
    
    return output


def analyze_sectors_with_mask(csv_filename: str, N_sector: List[int], years: np.ndarray) -> np.ndarray:
    """
    Version that pre-calculates angular masks based on 2*pi/N_sector[i].
    
    For each year and each N_sector configuration:
    - Pre-calculate angular width (2π/N_sectors)
    - Group points by angular position into sectors
    - Calculate mean and std for each angular sector
    - Average these statistics across all sectors
    
    Parameters:
    -----------
    csv_filename : str
        Path to CSV file containing sector data
    N_sector : List[int]
        List of sector counts (different angular resolutions to analyze)
    years : np.ndarray
        Array of years to analyze
    
    Returns:
    --------
    np.ndarray
        Array of shape (2*len(years), len(N_sector)) where:
        - Row 2*i contains averaged standard deviations for year i
        - Row 2*i+1 contains averaged mean radii for year i
        - Each column corresponds to one N_sector configuration
    """
    # Read CSV file
    df = pd.read_csv(csv_filename)
    
    # Initialize output array
    output = np.full((2 * len(years), len(N_sector)), np.nan)
    
    # Process each year
    for year_idx, year in enumerate(years):
        # Filter data for current year
        year_data = df[df['year'] == year]
        
        if len(year_data) == 0:
            continue
        
        # Extract distance data
        distances = year_data['distance_pixels'].values
        
        # Get actual angles
        if 'actual_angle_rad' in year_data.columns:
            angles = year_data['actual_angle_rad'].values
        elif 'sector_angle_rad' in year_data.columns:
            angles = year_data['sector_angle_rad'].values
        else:
            # Calculate angles from row/col if not available
            if 'row' in year_data.columns and 'col' in year_data.columns:
                center_row = year_data['center_row'].iloc[0] if 'center_row' in year_data.columns else year_data['row'].mean()
                center_col = year_data['center_col'].iloc[0] if 'center_col' in year_data.columns else year_data['col'].mean()
                
                dy = year_data['row'].values - center_row
                dx = year_data['col'].values - center_col
                angles = np.arctan2(dy, dx)
                # Normalize to [0, 2π)
                angles = np.where(angles < 0, angles + 2*np.pi, angles)
            else:
                continue
        
        # Process each N_sector configuration
        for n_idx, n_sectors in enumerate(N_sector):
            # Pre-calculate angular mask
            angular_width = 2 * np.pi / n_sectors
            
            # Calculate statistics using angular binning
            std_vals, mean_vals = compute_sector_stats_by_angle(
                angles, distances, n_sectors, angular_width
            )
            
            # Average over all sectors
            avg_std = np.nanmean(std_vals)
            avg_mean = np.nanmean(mean_vals)
            
            # Store results
            output[2 * year_idx, n_idx] = avg_std
            output[2 * year_idx + 1, n_idx] = avg_mean
    
    return output
def analyze_sectors_fast_vectorized(csv_filename: str, N_sector: List[int], years: np.ndarray) -> np.ndarray:
    """
    Highly optimized vectorized version using numpy for angular binning.
    
    This version uses vectorized operations throughout for maximum speed.
    
    Parameters:
    -----------
    csv_filename : str
        Path to CSV file containing sector data
    N_sector : List[int]
        List of sector counts (different angular resolutions to analyze)
    years : np.ndarray
        Array of years to analyze
    
    Returns:
    --------
    np.ndarray
        Array of shape (2*len(years), len(N_sector))
    """
    # Read CSV file once
    df = pd.read_csv(csv_filename)
    
    # Initialize output array
    output = np.full((2 * len(years), len(N_sector)), np.nan)
    
    # Process each year
    for year_idx, year in enumerate(years):
        # Filter data for current year
        year_data = df[df['year'] == year]
        
        if len(year_data) == 0:
            continue
        
        # Extract distances
        distances = year_data['distance_pixels'].values
        
        # Get angles
        if 'actual_angle_rad' in year_data.columns:
            angles = year_data['actual_angle_rad'].values
        elif 'sector_angle_rad' in year_data.columns:
            angles = year_data['sector_angle_rad'].values
        else:
            # Calculate angles from row/col
            if 'row' in year_data.columns and 'col' in year_data.columns:
                center_row = year_data['center_row'].iloc[0] if 'center_row' in year_data.columns else year_data['row'].mean()
                center_col = year_data['center_col'].iloc[0] if 'center_col' in year_data.columns else year_data['col'].mean()
                
                dy = year_data['row'].values - center_row
                dx = year_data['col'].values - center_col
                angles = np.arctan2(dy, dx)
                angles = np.where(angles < 0, angles + 2*np.pi, angles)
            else:
                continue
        
        # Process each N_sector configuration
        for n_idx, n_sectors in enumerate(N_sector):
            # Calculate angular width
            angular_width = 2 * np.pi / n_sectors
            
            # Assign points to sectors using vectorized operations
            sector_ids = np.floor(angles / angular_width).astype(int)
            sector_ids = np.clip(sector_ids, 0, n_sectors - 1)  # Handle edge case
            
            # Calculate statistics for each sector using vectorized operations
            sector_stds = np.zeros(n_sectors)
            sector_means = np.zeros(n_sectors)
            
            for sector_id in range(n_sectors):
                mask = sector_ids == sector_id
                sector_distances = distances[mask]
                
                if len(sector_distances) > 0:
                    sector_means[sector_id] = np.mean(sector_distances)
                    sector_stds[sector_id] = np.std(sector_distances)
                else:
                    sector_means[sector_id] = np.nan
                    sector_stds[sector_id] = np.nan
            
            # Average over all sectors
            output[2 * year_idx, n_idx] = np.nanmean(sector_stds)
            output[2 * year_idx + 1, n_idx] = np.nanmean(sector_means)
    
    return output


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Example parameters
    csv_file = "sector_data.csv"
    N_sector = [10, 20, 30, 40]  # Different angular resolutions to test
    years = np.array([1985, 1986, 1987, 1988])
    
    # Benchmark different approaches
    print("Benchmarking different implementations...")
    print(f"Testing {len(N_sector)} different sector configurations: {N_sector}")
    print(f"For {len(years)} years: {years}")
    print("\nNote: Points are grouped by angular position (k*2π/n to (k+1)*2π/n)")
    
    # Method 1: Numba optimized
    start = time.time()
    result1 = analyze_sectors_optimized(csv_file, N_sector, years)
    time1 = time.time() - start
    print(f"\nNumba optimized (angular binning): {time1:.4f} seconds")
    
    # Method 2: Vectorized pandas
    start = time.time()
    result2 = analyze_sectors_vectorized(csv_file, N_sector, years)
    time2 = time.time() - start
    print(f"Vectorized pandas (angular binning): {time2:.4f} seconds")
    
    # Method 3: With angular mask
    start = time.time()
    result3 = analyze_sectors_with_mask(csv_file, N_sector, years)
    time3 = time.time() - start
    print(f"With explicit angular mask: {time3:.4f} seconds")
    
    # Method 4: Fast vectorized
    start = time.time()
    result4 = analyze_sectors_fast_vectorized(csv_file, N_sector, years)
    time4 = time.time() - start
    print(f"Fast vectorized (numpy): {time4:.4f} seconds")
    
    print(f"\nOutput shape: {result1.shape} = (2*{len(years)} years, {len(N_sector)} configurations)")
    print(f"\nSample output (first year, all N_sector configurations):")
    print(f"Angular widths: {[f'{2*np.pi/n:.4f} rad ({np.degrees(2*np.pi/n):.1f}°)' for n in N_sector]}")
    print(f"Averaged Std devs: {result1[0, :]}")
    print(f"Averaged Means:    {result1[1, :]}")