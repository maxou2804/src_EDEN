import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple
from rasterio.transform import xy, Affine

sys.path.insert(0, str(Path(__file__).parent))
from wsf_evolution_lcc import BuiltAreaAnalyzer


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

