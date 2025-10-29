#!/usr/bin/env python3


import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path


def calculate_urban_density(wsf_data: np.ndarray,
                           year: int,
                           analyzer,  # BuiltAreaAnalyzer instance
                           radius_factor: float = 1.0) -> Dict:
    """
    Calculate urban density: ratio of LCC area to total urbanized area within a radius.
    
    The radius is defined as: radius_factor × mean_radius_of_LCC
    
    Args:
        wsf_data: WSF Evolution array
        year: Year to analyze (e.g., 1985)
        analyzer: BuiltAreaAnalyzer instance
        radius_factor: Multiplier for the LCC mean radius (default: 1.0)
                      e.g., 1.5 means analyze within 1.5× the LCC's mean radius
    
    Returns:
        Dictionary with:
            - year: Analysis year
            - lcc_area_km2: Area of largest connected component
            - lcc_pixels: Number of pixels in LCC
            - lcc_mean_radius_km: Mean radius of LCC (approximation)
            - search_radius_km: Radius used for density calculation
            - total_urban_area_km2: Total urbanized area within search radius
            - total_urban_pixels: Total urbanized pixels within search radius
            - urban_density: Ratio of LCC area to total urban area (0-1)
            - density_percentage: Urban density as percentage
    """
    pixel_area_km2 = (30 * 30) / 1e6  # 30m × 30m in km²
    
    print("\n" + "="*70)
    print(f"URBAN DENSITY CALCULATION - YEAR {year}")
    print("="*70)
    
    # Step 1: Extract mask for the given year
    print(f"\n1. Extracting urban areas for year {year}...")
    mask = analyzer.extract_year_mask(wsf_data, year)
    total_built_initial = mask.sum()
    print(f"   Total built pixels: {total_built_initial:,}")
    
    # Step 2: Find LCC
    print("\n2. Finding largest connected component...")
    lcc_mask, lcc_size = analyzer.find_largest_connected_component(mask)
    lcc_area_km2 = lcc_size * pixel_area_km2
    
    if lcc_size == 0:
        print("   ⚠️  No LCC found!")
        return {
            'year': year,
            'lcc_area_km2': 0,
            'lcc_pixels': 0,
            'lcc_mean_radius_km': 0,
            'search_radius_km': 0,
            'total_urban_area_km2': 0,
            'total_urban_pixels': 0,
            'urban_density': 0,
            'density_percentage': 0
        }
    
    print(f"   LCC pixels: {lcc_size:,}")
    print(f"   LCC area: {lcc_area_km2:.3f} km²")
    
    # Step 3: Calculate LCC mean radius (approximate as circle)
    # Area = π × r²  →  r = √(Area / π)
    lcc_mean_radius_km = np.sqrt(lcc_area_km2 / np.pi)
    print(f"\n3. LCC mean radius (as circle): {lcc_mean_radius_km:.3f} km")
    
    # Step 4: Calculate search radius
    search_radius_km = lcc_mean_radius_km * radius_factor
    print(f"\n4. Search radius ({radius_factor}× mean radius): {search_radius_km:.3f} km")
    
    # Step 5: Find LCC centroid
    print("\n5. Finding LCC centroid...")
    lcc_rows, lcc_cols = np.where(lcc_mask == 1)
    centroid_row = int(np.mean(lcc_rows))
    centroid_col = int(np.mean(lcc_cols))
    print(f"   Centroid pixel: ({centroid_row}, {centroid_col})")
    
    # Step 6: Create circular mask around centroid
    print(f"\n6. Creating circular search region ({search_radius_km:.3f} km radius)...")
    rows, cols = np.ogrid[0:wsf_data.shape[0], 0:wsf_data.shape[1]]
    
    # Convert radius from km to pixels (30m resolution)
    search_radius_pixels = search_radius_km * 1000 / 30
    
    # Distance from centroid in pixels
    distance_from_centroid = np.sqrt((rows - centroid_row)**2 + (cols - centroid_col)**2)
    
    # Circular mask
    circular_mask = distance_from_centroid <= search_radius_pixels
    print(f"   Search region covers {circular_mask.sum():,} pixels")
    
    # Step 7: Count total urban pixels within circular region
    print("\n7. Counting urban pixels within search region...")
    urban_within_radius = mask & circular_mask
    total_urban_pixels = urban_within_radius.sum()
    total_urban_area_km2 = total_urban_pixels * pixel_area_km2
    
    print(f"   Total urban pixels in region: {total_urban_pixels:,}")
    print(f"   Total urban area in region: {total_urban_area_km2:.3f} km²")
    
    # Step 8: Calculate density
    print("\n8. Calculating urban density...")
    if total_urban_area_km2 > 0:
        urban_density = lcc_area_km2 / total_urban_area_km2
        density_percentage = urban_density * 100
    else:
        urban_density = 0
        density_percentage = 0
    
    print(f"   Urban density: {density_percentage:.2f}%")
    print(f"   (LCC area / Total urban area within radius)")
    
    # Step 9: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Year: {year}")
    print(f"LCC Area: {lcc_area_km2:.3f} km² ({lcc_size:,} pixels)")
    print(f"LCC Mean Radius: {lcc_mean_radius_km:.3f} km")
    print(f"Search Radius: {search_radius_km:.3f} km ({radius_factor}× mean radius)")
    print(f"Total Urban Area in Region: {total_urban_area_km2:.3f} km² ({total_urban_pixels:,} pixels)")
    print(f"Urban Density: {density_percentage:.2f}%")
    print("="*70)
    
    return {
        'year': year,
        'lcc_area_km2': round(lcc_area_km2, 3),
        'lcc_pixels': int(lcc_size),
        'lcc_mean_radius_km': round(lcc_mean_radius_km, 3),
        'search_radius_km': round(search_radius_km, 3),
        'total_urban_area_km2': round(total_urban_area_km2, 3),
        'total_urban_pixels': int(total_urban_pixels),
        'urban_density': round(urban_density, 4),
        'density_percentage': round(density_percentage, 2)
    }


def calculate_urban_density_timeseries(wsf_data: np.ndarray,
                                      analyzer,  # BuiltAreaAnalyzer instance
                                      radius_factor: float = 1.0,
                                      years: List[int] = None) -> pd.DataFrame:
    """
    Calculate urban density for multiple years.
    
    Args:
        wsf_data: WSF Evolution array
        analyzer: BuiltAreaAnalyzer instance
        radius_factor: Multiplier for the LCC mean radius
        years: List of years to analyze (default: 1985-2015)
    
    Returns:
        DataFrame with density metrics for each year
    """
    if years is None:
        years = list(range(1985, 2016))
    
    results = []
    
    print("\n" + "="*70)
    print("URBAN DENSITY TIME SERIES ANALYSIS")
    print("="*70)
    print(f"Radius factor: {radius_factor}×")
    print(f"Years: {min(years)}-{max(years)}")
    print("="*70)
    
    for year in years:
        density_result = calculate_urban_density(wsf_data, year, analyzer, radius_factor)
        results.append(density_result)
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("TIME SERIES COMPLETE")
    print("="*70)
    print(f"Years analyzed: {len(df)}")
    print(f"Density range: {df['density_percentage'].min():.2f}% - {df['density_percentage'].max():.2f}%")
    print("="*70)
    
    return df








def visualize_density_calculation(wsf_data: np.ndarray,
                                  year: int,
                                  analyzer,  # BuiltAreaAnalyzer instance
                                  radius_factor: float = 1.0,
                                  output_path: str = None,
                                  zoom_factor: float = 1.15,
                                  n_clusters: int = 10,
                                  show_cluster_labels: bool = False):
    """
    Visualize clusters ONLY within the density calculation circle.
    
    Creates a clean figure showing:
    - Only clusters within the search radius
    - Each cluster in a different color
    - Search radius circle boundary
    - LCC centroid marker
    - No floating labels (clean view)
    
    Args:
        wsf_data: WSF Evolution array
        year: Year to visualize
        analyzer: BuiltAreaAnalyzer instance
        radius_factor: Multiplier for the LCC mean radius
        output_path: Optional path to save the figure
        zoom_factor: View extent multiplier (1.15 = 15% buffer)
        n_clusters: Number of top clusters to show with colors (default: 10)
        show_cluster_labels: Whether to show cluster rank labels (default: False)
    """
    import matplotlib.pyplot as plt
    from scipy import ndimage
    
    pixel_area_km2 = (30 * 30) / 1e6
    
    print(f"\nVisualizing density calculation for year {year}...")
    
    # Step 1: Get full urban mask
    mask = analyzer.extract_year_mask(wsf_data, year)
    
    if mask.sum() == 0:
        print("No urban areas found, cannot visualize")
        return
    
    # Step 2: Find LCC from full mask to get centroid and radius
    lcc_mask_full, lcc_size_full = analyzer.find_largest_connected_component(mask)
    
    if lcc_size_full == 0:
        print("No LCC found, cannot visualize")
        return
    
    lcc_area_km2 = lcc_size_full * pixel_area_km2
    lcc_mean_radius_km = np.sqrt(lcc_area_km2 / np.pi)
    search_radius_km = lcc_mean_radius_km * radius_factor
    search_radius_pixels = search_radius_km * 1000 / 30
    
    # Find LCC centroid
    lcc_rows, lcc_cols = np.where(lcc_mask_full == 1)
    centroid_row = int(np.mean(lcc_rows))
    centroid_col = int(np.mean(lcc_cols))
    
    print(f"  LCC: {lcc_area_km2:.1f} km² (radius: {lcc_mean_radius_km:.1f} km)")
    print(f"  Search radius: {search_radius_km:.1f} km ({radius_factor}× LCC radius)")
    
    # Step 3: Create circular mask
    rows, cols = np.ogrid[0:wsf_data.shape[0], 0:wsf_data.shape[1]]
    distance_from_centroid = np.sqrt((rows - centroid_row)**2 + (cols - centroid_col)**2)
    circular_mask = distance_from_centroid <= search_radius_pixels
    
    # Step 4: Apply circular mask to urban areas
    mask_in_circle = mask & circular_mask
    
    if mask_in_circle.sum() == 0:
        print("No urban areas found within search circle")
        return
    
    # Step 5: Find clusters ONLY within the circle
    labeled_array, num_features = ndimage.label(mask_in_circle)
    
    if num_features == 0:
        print("No connected components in search circle")
        return
    
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background
    
    # Get top N clusters
    n_to_show = min(n_clusters, len(component_sizes))
    top_indices = np.argsort(component_sizes)[-n_to_show:][::-1]  # Descending
    
    print(f"  Found {num_features} clusters in circle, showing top {n_to_show}")
    
    # Step 6: Create color map
    if n_to_show <= 10:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(n_to_show)]
    elif n_to_show <= 20:
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in range(n_to_show)]
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_to_show))
    
    # Step 7: Create RGB image - BLACK background
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Color each cluster
    cluster_info = []
    for i, component_idx in enumerate(top_indices):
        component_label = component_idx + 1
        component_mask = (labeled_array == component_label)
        
        # Apply color
        color_rgb = (np.array(colors[i][:3]) * 255).astype(np.uint8)
        rgb[component_mask] = color_rgb
        
        area_km2 = component_sizes[component_idx] * pixel_area_km2
        
        # Get centroid
        c_rows, c_cols = np.where(component_mask)
        if len(c_rows) > 0:
            c_centroid_row = int(np.mean(c_rows))
            c_centroid_col = int(np.mean(c_cols))
        else:
            c_centroid_row = c_centroid_col = 0
        
        cluster_info.append({
            'rank': i + 1,
            'label': "LCC" if i == 0 else f"C{i+1}",
            'area_km2': area_km2,
            'centroid': (c_centroid_row, c_centroid_col),
            'color': colors[i]
        })
    
    # Calculate density
    total_urban_area_km2 = sum(info['area_km2'] for info in cluster_info)
    lcc_area_in_circle = cluster_info[0]['area_km2']
    density_percentage = (lcc_area_in_circle / total_urban_area_km2 * 100) if total_urban_area_km2 > 0 else 0
    
    # Step 8: Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Show the image
    ax.imshow(rgb)
    
    # Draw search radius circle
    circle = plt.Circle((centroid_col, centroid_row), 
                       search_radius_pixels,
                       fill=False, edgecolor='white', linewidth=3,
                       linestyle='--', alpha=0.9, zorder=10)
    ax.add_patch(circle)
    
    # Mark LCC centroid
    ax.plot(centroid_col, centroid_row, '*', 
           color='yellow', markersize=30,
           markeredgecolor='white', markeredgewidth=2, 
           zorder=11)
    
    # Optional: Add labels at centroids
    if show_cluster_labels:
        for info in cluster_info[:5]:  # Only label top 5 to avoid clutter
            c_row, c_col = info['centroid']
            ax.text(c_col, c_row, info['label'],
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='circle', facecolor='black', 
                            alpha=0.7, edgecolor='white', linewidth=1),
                   zorder=9)
    
    # Step 9: Set view to show circle with padding
    view_extent = search_radius_pixels * zoom_factor
    
    xlim_min = max(0, centroid_col - view_extent)
    xlim_max = min(wsf_data.shape[1], centroid_col + view_extent)
    ylim_min = max(0, centroid_row - view_extent)
    ylim_max = min(wsf_data.shape[0], centroid_row + view_extent)
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_max, ylim_min)  # Inverted for image coordinates
    
    # Step 10: Title
    ax.set_title(
        f'Urban Clusters Within Density Circle - {year}\n'
        f'Search Radius: {search_radius_km:.1f} km ({radius_factor}× LCC radius) | '
        f'Density: {density_percentage:.1f}% | '
        f'Clusters: {n_to_show}',
        fontsize=14, fontweight='bold', pad=15,
        color='black'
    )
    
    # Step 11: Legend - COMPACT and CLEAR
    from matplotlib.patches import Patch
    legend_elements = []
    
    # Top clusters (max 10 in legend)
    for info in cluster_info[:min(10, len(cluster_info))]:
        label_text = f"{info['label']}: {info['area_km2']:.1f} km²"
        legend_elements.append(
            Patch(facecolor=info['color'], edgecolor='white', linewidth=0.5,
                  label=label_text)
        )
    
    if len(cluster_info) > 10:
        legend_elements.append(
            Patch(facecolor='gray', alpha=0.5,
                  label=f'+ {len(cluster_info)-10} more')
        )
    
    # Add separator
    legend_elements.append(
        Patch(facecolor='none', edgecolor='none', label='')
    )
    
    # Add circle and centroid info
    legend_elements.append(
        plt.Line2D([0], [0], color='white', linewidth=3, linestyle='--',
                  label=f'Search circle')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='*', color='w', 
                  markerfacecolor='yellow', markersize=15,
                  markeredgecolor='white', markeredgewidth=2,
                  linestyle='None', label='LCC center')
    )
    
    ax.legend(handles=legend_elements, 
             loc='upper right',
             fontsize=10,
             framealpha=0.95,
             edgecolor='black',
             fancybox=True,
             title='Clusters by Size',
             title_fontsize=11)
    
    ax.axis('off')
    
    # Add subtle grid in background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {output_path}")
        print(f"  Resolution: 200 DPI")
        print(f"  View: {search_radius_km:.1f} km radius")
        print(f"  Clusters shown: {n_to_show}")
        print(f"  Density: {density_percentage:.1f}%")
    else:
        plt.show()
    
    return fig

def track_urban_clusters_growth(wsf_data: np.ndarray,
                                analyzer,  # BuiltAreaAnalyzer instance
                                n_clusters: int = 5,
                                start_year: int = 1985,
                                end_year: int = 2015) -> Dict:
    """
    Track the growth rate of the largest urban cluster (LCC) and the next N largest clusters.
    
    This function identifies urban clusters and tracks their evolution over time,
    calculating growth rates for each cluster.
    
    Args:
        wsf_data: WSF Evolution array
        analyzer: BuiltAreaAnalyzer instance
        n_clusters: Number of additional clusters to track (beyond the LCC)
                   Default: 5 (tracks LCC + 5 next largest = 6 total clusters)
        start_year: Start year for analysis (default: 1985)
        end_year: End year for analysis (default: 2015)
    
    Returns:
        Dictionary with:
            - 'summary': DataFrame with growth statistics for each cluster
            - 'timeseries': DataFrame with year-by-year data for all clusters
            - 'metadata': Additional information about the analysis
    """
    import pandas as pd
    from scipy import ndimage
    
    pixel_area_km2 = (30 * 30) / 1e6  # 30m × 30m in km²
    years = list(range(start_year, end_year + 1))
    
    print("\n" + "="*70)
    print(f"TRACKING TOP {n_clusters + 1} URBAN CLUSTERS GROWTH")
    print("="*70)
    print(f"Period: {start_year}-{end_year}")
    print(f"Tracking: LCC + {n_clusters} next largest clusters")
    print("="*70)
    
    # Data structure to store cluster information
    cluster_data = []
    
    # Process each year
    for year_idx, year in enumerate(years):
        if year_idx % 5 == 0 or year == years[0] or year == years[-1]:
            print(f"\nProcessing year {year}...")
        
        # Extract mask for this year
        mask = analyzer.extract_year_mask(wsf_data, year)
        
        if mask.sum() == 0:
            print(f"  Warning: No urban areas found in {year}")
            continue
        
        # Label all connected components
        labeled_array, num_features = ndimage.label(mask)
        
        if num_features == 0:
            print(f"  Warning: No connected components in {year}")
            continue
        
        # Get size of each component
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
        
        # Get top N+1 components (LCC + N others)
        n_to_track = min(n_clusters + 1, len(component_sizes))
        top_indices = np.argsort(component_sizes)[-n_to_track:][::-1]  # Descending order
        
        # Store data for each cluster
        for rank, component_idx in enumerate(top_indices):
            component_label = component_idx + 1  # Labels start from 1
            size_pixels = component_sizes[component_idx]
            area_km2 = size_pixels * pixel_area_km2
            
            # Get component mask
            component_mask = (labeled_array == component_label)
            
            # Calculate centroid
            rows, cols = np.where(component_mask)
            if len(rows) > 0:
                centroid_row = float(np.mean(rows))
                centroid_col = float(np.mean(cols))
            else:
                centroid_row = centroid_col = 0
            
            cluster_data.append({
                'year': year,
                'rank': rank + 1,  # 1 = LCC, 2 = second largest, etc.
                'cluster_label': f"Cluster_{rank + 1}" if rank > 0 else "LCC",
                'size_pixels': int(size_pixels),
                'area_km2': round(area_km2, 3),
                'centroid_row': round(centroid_row, 1),
                'centroid_col': round(centroid_col, 1),
                'num_total_clusters': num_features
            })
    
    # Convert to DataFrame
    df_timeseries = pd.DataFrame(cluster_data)
    
    print("\n" + "="*70)
    print("CALCULATING GROWTH RATES")
    print("="*70)
    
    # Calculate growth statistics for each cluster
    summary_data = []
    
    for rank in range(1, n_clusters + 2):  # 1 to n_clusters+1
        cluster_name = "LCC" if rank == 1 else f"Cluster_{rank}"
        cluster_df = df_timeseries[df_timeseries['rank'] == rank].sort_values('year')
        
        if len(cluster_df) < 2:
            print(f"  {cluster_name}: Insufficient data")
            continue
        
        # Get start and end values
        first_year = cluster_df.iloc[0]
        last_year = cluster_df.iloc[-1]
        
        start_area = first_year['area_km2']
        end_area = last_year['area_km2']
        
        # Calculate growth metrics
        absolute_growth = end_area - start_area
        percent_growth = ((end_area - start_area) / start_area * 100) if start_area > 0 else 0
        
        # Calculate annual growth rate (CAGR - Compound Annual Growth Rate)
        n_years = last_year['year'] - first_year['year']
        if n_years > 0 and start_area > 0:
            cagr = (((end_area / start_area) ** (1 / n_years)) - 1) * 100
        else:
            cagr = 0
        
        # Calculate average annual absolute growth
        avg_annual_growth = absolute_growth / n_years if n_years > 0 else 0
        
        # Check if cluster appears in all years
        years_present = len(cluster_df)
        total_years = len(years)
        persistence = (years_present / total_years * 100)
        
        summary_data.append({
            'rank': rank,
            'cluster_name': cluster_name,
            'start_year': int(first_year['year']),
            'end_year': int(last_year['year']),
            'start_area_km2': round(start_area, 3),
            'end_area_km2': round(end_area, 3),
            'absolute_growth_km2': round(absolute_growth, 3),
            'percent_growth': round(percent_growth, 2),
            'cagr_percent': round(cagr, 2),
            'avg_annual_growth_km2': round(avg_annual_growth, 3),
            'years_present': years_present,
            'total_years': total_years,
            'persistence_percent': round(persistence, 2)
        })
        
        print(f"\n{cluster_name}:")
        print(f"  {first_year['year']}: {start_area:.2f} km² → {last_year['year']}: {end_area:.2f} km²")
        print(f"  Growth: {absolute_growth:.2f} km² ({percent_growth:.1f}%)")
        print(f"  CAGR: {cagr:.2f}% per year")
        print(f"  Present in {years_present}/{total_years} years ({persistence:.1f}%)")
    
    df_summary = pd.DataFrame(summary_data)
    
    # Prepare metadata
    metadata = {
        'start_year': start_year,
        'end_year': end_year,
        'n_years': end_year - start_year,
        'n_clusters_tracked': n_clusters + 1,
        'pixel_resolution_m': 30,
        'pixel_area_km2': pixel_area_km2
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Period analyzed: {start_year}-{end_year} ({end_year - start_year} years)")
    print(f"Clusters tracked: {len(summary_data)}")
    print(f"\nLCC Growth:")
    if len(df_summary) > 0:
        lcc_row = df_summary[df_summary['rank'] == 1].iloc[0]
        print(f"  Area: {lcc_row['start_area_km2']} → {lcc_row['end_area_km2']} km²")
        print(f"  Total growth: {lcc_row['absolute_growth_km2']} km² ({lcc_row['percent_growth']}%)")
        print(f"  CAGR: {lcc_row['cagr_percent']}%")
    print("="*70)
    
    return {
        'summary': df_summary,
        'timeseries': df_timeseries,
        'metadata': metadata
    }


def visualize_clusters_growth(growth_results: Dict,
                              output_path: str = None,
                              show_top_n: int = None):
    """
    Create visualizations of cluster growth over time.
    
    Args:
        growth_results: Output from track_urban_clusters_growth()
        output_path: Optional path to save the figure
        show_top_n: Optional limit on number of clusters to show (default: all)
    """
    import matplotlib.pyplot as plt
    
    df_summary = growth_results['summary']
    df_timeseries = growth_results['timeseries']
    
    if show_top_n is not None:
        df_summary = df_summary.head(show_top_n)
        ranks_to_show = df_summary['rank'].tolist()
        df_timeseries = df_timeseries[df_timeseries['rank'].isin(ranks_to_show)]
    
    n_clusters = len(df_summary)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    # Plot 1: Area over time (line plot)
    ax1 = fig.add_subplot(gs[0, :])
    for idx, (rank, cluster_name) in enumerate(zip(df_summary['rank'], df_summary['cluster_name'])):
        cluster_ts = df_timeseries[df_timeseries['rank'] == rank].sort_values('year')
        ax1.plot(cluster_ts['year'], cluster_ts['area_km2'], 
                marker='o', linewidth=2, label=cluster_name, 
                color=colors[idx], markersize=4)
    
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Area (km²)', fontsize=11)
    ax1.set_title('Urban Cluster Growth Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute growth (bar chart)
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.barh(df_summary['cluster_name'], df_summary['absolute_growth_km2'], 
                     color=colors)
    ax2.set_xlabel('Absolute Growth (km²)', fontsize=11)
    ax2.set_title('Total Area Growth', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_summary['absolute_growth_km2'])):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Plot 3: Percent growth (bar chart)
    ax3 = fig.add_subplot(gs[1, 1])
    bars = ax3.barh(df_summary['cluster_name'], df_summary['percent_growth'], 
                     color=colors)
    ax3.set_xlabel('Percent Growth (%)', fontsize=11)
    ax3.set_title('Relative Growth Rate', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_summary['percent_growth'])):
        ax3.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Plot 4: CAGR comparison (bar chart)
    ax4 = fig.add_subplot(gs[2, 0])
    bars = ax4.barh(df_summary['cluster_name'], df_summary['cagr_percent'], 
                     color=colors)
    ax4.set_xlabel('CAGR (%/year)', fontsize=11)
    ax4.set_title('Compound Annual Growth Rate', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_summary['cagr_percent'])):
        ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Plot 5: Persistence (how often cluster appears)
    ax5 = fig.add_subplot(gs[2, 1])
    bars = ax5.barh(df_summary['cluster_name'], df_summary['persistence_percent'], 
                     color=colors)
    ax5.set_xlabel('Persistence (%)', fontsize=11)
    ax5.set_title('Temporal Persistence', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 105)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_summary['persistence_percent'])):
        ax5.text(val, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Urban Clusters Growth Analysis ({growth_results["metadata"]["start_year"]}-{growth_results["metadata"]["end_year"]})',
                 fontsize=15, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Growth visualization saved: {output_path}")
    else:
        plt.show()


def export_clusters_growth_report(growth_results: Dict,
                                  output_dir: str = "./",
                                  city_name: str = "Unknown"):
    """
    Export comprehensive growth analysis report to CSV files.
    
    Args:
        growth_results: Output from track_urban_clusters_growth()
        output_dir: Directory to save output files
        city_name: Name of the city for file naming
    
    Returns:
        Dictionary with paths to saved files
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Sanitize city name for filename
    safe_city_name = "".join(c for c in city_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_city_name = safe_city_name.replace(' ', '_')
    
    files_created = {}
    
    # 1. Save summary statistics
    summary_path = output_dir / f"{safe_city_name}_clusters_growth_summary.csv"
    growth_results['summary'].to_csv(summary_path, index=False)
    files_created['summary'] = str(summary_path)
    print(f"✓ Saved summary: {summary_path}")
    
    # 2. Save time series data
    timeseries_path = output_dir / f"{safe_city_name}_clusters_timeseries.csv"
    growth_results['timeseries'].to_csv(timeseries_path, index=False)
    files_created['timeseries'] = str(timeseries_path)
    print(f"✓ Saved time series: {timeseries_path}")
    
    # 3. Save metadata
    metadata_path = output_dir / f"{safe_city_name}_analysis_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("URBAN CLUSTERS GROWTH ANALYSIS - METADATA\n")
        f.write("=" * 60 + "\n\n")
        for key, value in growth_results['metadata'].items():
            f.write(f"{key}: {value}\n")
    files_created['metadata'] = str(metadata_path)
    print(f"✓ Saved metadata: {metadata_path}")
    
    return files_created


def visualize_clusters_map(wsf_data: np.ndarray,
                           year: int,
                           analyzer,  # BuiltAreaAnalyzer instance
                           n_clusters: int = 10,
                           figsize: tuple = (12, 10),
                           center_on_lcc: bool = True,
                           zoom_factor: float = 1.2,
                           show_labels: bool = True,
                           output_path: str = None):
    """
    Visualize clusters on the city map with different colors.
    
    Args:
        wsf_data: WSF Evolution array
        year: Year to visualize
        analyzer: BuiltAreaAnalyzer instance
        n_clusters: Number of top clusters to show (default: 10)
        figsize: Figure size (width, height)
        center_on_lcc: Whether to center view on LCC (default: True)
        zoom_factor: View extent multiplier if centering (default: 1.2)
        show_labels: Whether to show cluster labels (default: True)
        output_path: Optional path to save the figure
    
    Returns:
        Figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy import ndimage
    
    print(f"\nVisualizing clusters for year {year}...")
    
    # Extract mask for this year
    mask = analyzer.extract_year_mask(wsf_data, year)
    
    if mask.sum() == 0:
        print(f"  Warning: No urban areas found in {year}")
        return None
    
    # Label all connected components
    labeled_array, num_features = ndimage.label(mask)
    
    if num_features == 0:
        print(f"  Warning: No connected components in {year}")
        return None
    
    # Get size of each component
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background
    
    # Get top N clusters
    n_to_show = min(n_clusters, len(component_sizes))
    top_indices = np.argsort(component_sizes)[-n_to_show:][::-1]  # Descending
    
    print(f"  Found {num_features} clusters, showing top {n_to_show}")
    
    # Create color map for clusters
    # Use distinguishable colors
    if n_to_show <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_to_show]
    elif n_to_show <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_to_show]
    else:
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_to_show))
    
    # Create RGB image
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Color each cluster
    cluster_info = []
    for i, component_idx in enumerate(top_indices):
        component_label = component_idx + 1
        component_mask = (labeled_array == component_label)
        
        # Apply color
        rgb[component_mask] = (colors[i, :3] * 255).astype(np.uint8)
        
        # Get centroid for label
        rows, cols = np.where(component_mask)
        centroid_row = int(np.mean(rows))
        centroid_col = int(np.mean(cols))
        
        area_km2 = component_sizes[component_idx] * (30 * 30) / 1e6
        
        cluster_info.append({
            'rank': i + 1,
            'label': f"#{i+1}" if i > 0 else "LCC",
            'size_pixels': component_sizes[component_idx],
            'area_km2': area_km2,
            'centroid': (centroid_row, centroid_col),
            'color': colors[i]
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show the image
    ax.imshow(rgb)
    
    # Add labels if requested
    if show_labels:
        for info in cluster_info:
            centroid_row, centroid_col = info['centroid']
            label_text = info['label']
            
            # Use white or black text depending on background brightness
            brightness = np.mean(info['color'][:3])
            text_color = 'white' if brightness < 0.5 else 'black'
            
            ax.plot(centroid_col, centroid_row, 'o', 
                   color='white', markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            ax.text(centroid_col, centroid_row, label_text,
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color=text_color)
    
    # Center view on LCC if requested
    if center_on_lcc and len(cluster_info) > 0:
        lcc_centroid = cluster_info[0]['centroid']
        lcc_size = cluster_info[0]['size_pixels']
        
        # Estimate LCC radius
        lcc_radius_pixels = np.sqrt(lcc_size / np.pi)
        view_extent = lcc_radius_pixels * zoom_factor
        
        # Calculate view bounds
        xlim_min = max(0, lcc_centroid[1] - view_extent)
        xlim_max = min(wsf_data.shape[1], lcc_centroid[1] + view_extent)
        ylim_min = max(0, lcc_centroid[0] - view_extent)
        ylim_max = min(wsf_data.shape[0], lcc_centroid[0] + view_extent)
        
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_max, ylim_min)  # Inverted for image coordinates
    
    # Title with statistics
    total_urban_area = sum(info['area_km2'] for info in cluster_info)
    lcc_area = cluster_info[0]['area_km2']
    lcc_dominance = (lcc_area / total_urban_area * 100) if total_urban_area > 0 else 0
    
    ax.set_title(f'Urban Clusters Map - Year {year}\n'
                f'Total Clusters: {num_features} | '
                f'Showing Top: {n_to_show} | '
                f'LCC: {lcc_area:.1f} km² ({lcc_dominance:.1f}%)',
                fontsize=13, fontweight='bold', pad=10)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = []
    for info in cluster_info[:min(10, len(cluster_info))]:  # Show max 10 in legend
        label = f"{info['label']}: {info['area_km2']:.1f} km²"
        legend_elements.append(Patch(facecolor=info['color'], label=label))
    
    if len(cluster_info) > 10:
        legend_elements.append(Patch(facecolor='gray', alpha=0.3, 
                                     label=f'... +{len(cluster_info)-10} more'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
             framealpha=0.9, title='Clusters (by size)')
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return fig

def create_clusters_animation(wsf_data: np.ndarray,
                              analyzer,  # BuiltAreaAnalyzer instance
                              output_path: str = "clusters_animation.gif",
                              start_year: int = 1985,
                              end_year: int = 2015,
                              n_clusters: int = 10,
                              figsize: tuple = (12, 10),
                              center_on_lcc: bool = True,
                              zoom_factor: float = 1.2,
                              show_labels: bool = True,
                              fps: int = 2,
                              keep_frames: bool = False):
    """
    Create an animated GIF showing cluster evolution over time.
    
    Args:
        wsf_data: WSF Evolution array
        analyzer: BuiltAreaAnalyzer instance
        output_path: Path to save the GIF file
        start_year: Start year (default: 1985)
        end_year: End year (default: 2015)
        n_clusters: Number of top clusters to show
        figsize: Figure size (width, height)
        center_on_lcc: Whether to center view on LCC
        zoom_factor: View extent multiplier
        show_labels: Whether to show cluster labels
        fps: Frames per second (default: 2)
        keep_frames: Whether to keep individual frame images (default: False)
    
    Returns:
        Path to the created GIF file
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    
    print("\n" + "="*70)
    print("CREATING CLUSTERS ANIMATION")
    print("="*70)
    print(f"Period: {start_year}-{end_year}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    print("="*70)
    
    years = list(range(start_year, end_year + 1))
    temp_dir = Path("./temp_animation_frames")
    temp_dir.mkdir(exist_ok=True)
    
    frame_paths = []
    
    # Generate frames
    for idx, year in enumerate(years, 1):
        print(f"\nGenerating frame {idx}/{len(years)}: Year {year}...")
        
        frame_path = temp_dir / f"frame_{year}.png"
        
        fig = visualize_clusters_map(
            wsf_data=wsf_data,
            year=year,
            analyzer=analyzer,
            n_clusters=n_clusters,
            figsize=figsize,
            center_on_lcc=center_on_lcc,
            zoom_factor=zoom_factor,
            show_labels=show_labels,
            output_path=str(frame_path)
        )
        
        if fig is not None:
            plt.close(fig)
            frame_paths.append(frame_path)
        else:
            print(f"  Warning: Could not create frame for {year}")
    
    # Create GIF
    if len(frame_paths) == 0:
        print("\n✗ Error: No frames were created!")
        return None
    
    print(f"\n{'='*70}")
    print("CREATING GIF")
    print("="*70)
    print(f"Loading {len(frame_paths)} frames...")
    
    images = []
    for frame_path in frame_paths:
        img = Image.open(frame_path)
        images.append(img)
    
    # Calculate duration in milliseconds
    duration_ms = int(1000 / fps)
    
    print(f"Saving GIF to {output_path}...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,  # Loop forever
        optimize=False
    )
    
    print(f"✓ GIF created: {output_path}")
    
    # Get file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Frames: {len(images)}")
    print(f"  Duration per frame: {duration_ms} ms")
    
    # Clean up temporary files
    if not keep_frames:
        print("\nCleaning up temporary frames...")
        for frame_path in frame_paths:
            frame_path.unlink()
        temp_dir.rmdir()
        print("  ✓ Temporary files removed")
    else:
        print(f"\n✓ Frames saved in: {temp_dir}")
    
    print("="*70)
    
    return output_path
