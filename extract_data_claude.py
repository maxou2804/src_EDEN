#!/usr/bin/env python3
"""
Complete WSF Evolution Analysis Workflow
=========================================

Integrates WSFTileManager (downloads tiles) with BuiltAreaAnalyzer (analyzes data)

This script:
1. Downloads all tiles needed for a city
2. Mosaics multiple tiles automatically
3. Extracts region of interest
4. Analyzes urban evolution (1985-2015)
5. Calculates LCC, perimeter, and other metrics
6. Saves results to DataFrame
7. Creates visualizations

Usage:
    python complete_wsf_workflow.py --city "Beijing, China" --radius 50 --output beijing_results
    python complete_wsf_workflow.py --lat 31.8122 --lon 119.9692 --radius 25 --output changzhou
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from geopy.geocoders import Nominatim
from typing import Dict
import sys

# Import the two main classes
sys.path.insert(0, str(Path(__file__).parent))
from wsf_evolution_lcc import WSFTileManager
from wsf_evolution_lcc import BuiltAreaAnalyzer


def geocode_city(city_name: str):
    """Geocode city name to coordinates"""
    print(f"Geocoding: {city_name}")
    geolocator = Nominatim(user_agent="wsf_complete_workflow")
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            print(f"Found: {location.address}\n")
            return location.latitude, location.longitude
        raise ValueError(f"Could not geocode: {city_name}")
    except Exception as e:
        raise ValueError(f"Geocoding failed: {e}")


def complete_workflow(center_lat: float, center_lon: float, 
                      radius_km: float, output_dir: str,
                      location_name: str = None,
                      cache_dir: str = "./wsf_cache",
                      analysis_size_km: float = None):
    """
    Complete workflow: Download tiles → Analyze → Save results
    
    Args:
        center_lat, center_lon: Center coordinates
        radius_km: Radius for tile download
        output_dir: Directory for results
        location_name: Name for output files
        cache_dir: Cache directory for tiles
        analysis_size_km: Size for analysis (if different from radius*2)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if location_name is None:
        location_name = f"location_{center_lat:.2f}_{center_lon:.2f}"
    
    # Clean location name for filenames
    safe_name = location_name.replace(" ", "_").replace(",", "")
    
    # Default analysis size is 2x radius
    if analysis_size_km is None:
        analysis_size_km = radius_km * 2
    
    print("="*80)
    print("COMPLETE WSF EVOLUTION ANALYSIS WORKFLOW")
    print("="*80)
    print(f"\nLocation: {location_name}")
    print(f"Center: ({center_lat:.4f}, {center_lon:.4f})")
    print(f"Download radius: {radius_km} km")
    print(f"Analysis size: {analysis_size_km} km × {analysis_size_km} km")
    print(f"Output directory: {output_path}")
    print(f"Cache directory: {cache_dir}\n")
    
    # =========================================================================
    # STEP 1: DOWNLOAD TILES
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: DOWNLOADING TILES")
    print("="*80)
    
    tile_manager = WSFTileManager(cache_dir=cache_dir)
    download_result = tile_manager.download_region(center_lat, center_lon, radius_km)
    
    # Check if we got any tiles
    downloaded_count = sum(1 for t in download_result['tiles'] if t['downloaded'])
    if downloaded_count == 0:
        print("\n✗ ERROR: No tiles were downloaded!")
        print("This region may not be covered by WSF Evolution dataset.")
        return None
    
    print(f"\n✓ Successfully downloaded/cached {downloaded_count} tile(s)")
    
    # Save download manifest
    manifest_file = output_path / f"{safe_name}_tile_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(download_result, f, indent=2)
    print(f"✓ Tile manifest saved: {manifest_file}")
    
    # =========================================================================
    # STEP 2: CREATE COVERAGE MAP
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: CREATING COVERAGE MAP")
    print("="*80)
    
    coverage_map_file = output_path / f"{safe_name}_coverage_map.png"
    tile_manager.visualize_coverage(download_result, output_file=str(coverage_map_file))
    
    # =========================================================================
    # STEP 3: LOAD AND MOSAIC TILES
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: LOADING AND MOSAICKING TILES")
    print("="*80)
    
    analyzer = BuiltAreaAnalyzer()
    
    try:
        full_data, full_metadata = analyzer.load_tiles_from_download_result(download_result)
        print(f"✓ Loaded mosaic: {full_data.shape[0]} × {full_data.shape[1]} pixels")
        print(f"  Coverage: {full_metadata['num_tiles']} tile(s)")
    except Exception as e:
        print(f"\n✗ ERROR loading tiles: {e}")
        return None
    
    # =========================================================================
    # STEP 4: EXTRACT REGION OF INTEREST
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: EXTRACTING REGION OF INTEREST")
    print("="*80)
    
    analysis_data, analysis_metadata = analyzer.extract_built_area_bbox(
        data=full_data,
        transform=full_metadata['transform'],
        center_lat=center_lat,
        center_lon=center_lon,
        size_km=analysis_size_km
    )
    
    print(f"✓ Extracted {analysis_size_km}km × {analysis_size_km}km region")
    print(f"  Size: {analysis_data.shape[0]} × {analysis_data.shape[1]} pixels")
    
    # =========================================================================
    # STEP 5: ANALYZE URBAN EVOLUTION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: ANALYZING URBAN EVOLUTION (1985-2015)")
    print("="*80)
    
    results = analyzer.analyze_evolution(analysis_data)
    
    # =========================================================================
    # STEP 6: CREATE DATAFRAME AND SAVE
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAVING RESULTS")
    print("="*80)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add metadata columns
    df['location'] = location_name
    df['center_lat'] = center_lat
    df['center_lon'] = center_lon
    df['analysis_size_km'] = analysis_size_km
    
    # Calculate additional derived metrics
    df['lcc_density'] = df['lcc_pixels'] / (analysis_data.shape[0] * analysis_data.shape[1])
    df['compactness'] = (4 * np.pi * df['lcc_area_km2']) / (df['perimeter_km'] ** 2)
    
    # Save to CSV
    csv_file = output_path / f"{safe_name}_timeseries.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ CSV saved: {csv_file}")
    
    # Save to pickle (faster loading)
    pkl_file = output_path / f"{safe_name}_timeseries.pkl"
    df.to_pickle(pkl_file)
    print(f"✓ Pickle saved: {pkl_file}")
    
    # =========================================================================
    # STEP 7: CREATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: CREATING VISUALIZATIONS")
    print("="*80)
    
    # Evolution visualization
    evolution_file = output_path / f"{safe_name}_evolution.png"
    analyzer.visualize_evolution(analysis_data, output_path=str(evolution_file))
    
    # Metrics plots
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: LCC Area over time
    axes[0, 0].plot(df['years'], df['lcc_area_km2'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('LCC Area (km²)')
    axes[0, 0].set_title('Urban Growth (Largest Connected Component)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Perimeter over time
    axes[0, 1].plot(df['years'], df['perimeter_km'], 'o-', color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Perimeter (km)')
    axes[0, 1].set_title('Urban Perimeter Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Number of components
    axes[1, 0].plot(df['years'], df['num_components'], 'o-', color='green', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Number of Components')
    axes[1, 0].set_title('Urban Fragmentation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Compactness
    axes[1, 1].plot(df['years'], df['compactness'], 'o-', color='red', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Compactness')
    axes[1, 1].set_title('Urban Compactness (4πA/P²)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.suptitle(f'Urban Evolution Metrics: {location_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    metrics_file = output_path / f"{safe_name}_metrics.png"
    plt.savefig(metrics_file, dpi=150, bbox_inches='tight')
    print(f"✓ Metrics plot saved: {metrics_file}")
    plt.close()
    
    # =========================================================================
    # STEP 8: GENERATE SUMMARY REPORT
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary_file = output_path / f"{safe_name}_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WSF EVOLUTION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Location: {location_name}\n")
        f.write(f"Center: ({center_lat:.4f}, {center_lon:.4f})\n")
        f.write(f"Analysis area: {analysis_size_km} km × {analysis_size_km} km\n")
        f.write(f"Period: 1985-2015\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KEY METRICS (2015)\n")
        f.write("-"*80 + "\n")
        last_row = df.iloc[-1]
        f.write(f"LCC Area:           {last_row['lcc_area_km2']:.2f} km²\n")
        f.write(f"LCC Pixels:         {last_row['lcc_pixels']:,}\n")
        f.write(f"Perimeter:          {last_row['perimeter_km']:.1f} km\n")
        f.write(f"Perimeter (pixels): {last_row['perimeter_pixels']:,}\n")
        f.write(f"Components:         {last_row['num_components']}\n")
        f.write(f"Compactness:        {last_row['compactness']:.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("GROWTH STATISTICS\n")
        f.write("-"*80 + "\n")
        first_row = df.iloc[0]
        area_growth = last_row['lcc_area_km2'] - first_row['lcc_area_km2']
        area_growth_pct = (area_growth / first_row['lcc_area_km2'] * 100) if first_row['lcc_area_km2'] > 0 else 0
        
        f.write(f"Initial area (1985): {first_row['lcc_area_km2']:.2f} km²\n")
        f.write(f"Final area (2015):   {last_row['lcc_area_km2']:.2f} km²\n")
        f.write(f"Growth:              {area_growth:.2f} km² ({area_growth_pct:.1f}%)\n")
        f.write(f"Annual growth rate:  {area_growth/30:.2f} km²/year\n\n")
        
        f.write("-"*80 + "\n")
        f.write("YEAR-BY-YEAR DATA\n")
        f.write("-"*80 + "\n\n")
        f.write(df[['years', 'lcc_area_km2', 'perimeter_km', 'num_components']].to_string(index=False))
        f.write("\n\n")
        
        f.write("-"*80 + "\n")
        f.write("FILES GENERATED\n")
        f.write("-"*80 + "\n")
        f.write(f"Timeseries data: {csv_file.name}\n")
        f.write(f"Pickle file: {pkl_file.name}\n")
        f.write(f"Evolution viz: {evolution_file.name}\n")
        f.write(f"Metrics plot: {metrics_file.name}\n")
        f.write(f"Coverage map: {coverage_map_file.name}\n")
        f.write(f"Tile manifest: {manifest_file.name}\n")
        f.write(f"This summary: {summary_file.name}\n")
    
    print(f"✓ Summary report saved: {summary_file}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETE!")
    print("="*80)
    
    print(f"\n📊 KEY RESULTS FOR 2015:")
    print(f"  LCC Area:     {last_row['lcc_area_km2']:.2f} km²")
    print(f"  Perimeter:    {last_row['perimeter_km']:.1f} km ({last_row['perimeter_pixels']:,} pixels)")
    print(f"  Components:   {last_row['num_components']}")
    print(f"  Compactness:  {last_row['compactness']:.4f}")
    
    print(f"\n📁 All results saved to: {output_path}/")
    print(f"\n💡 Next steps:")
    print(f"  1. View summary: cat {summary_file}")
    print(f"  2. View metrics: open {metrics_file}")
    print(f"  3. Load data: df = pd.read_csv('{csv_file}')")
    print(f"  4. Critical exponents: python critical_exponent_analysis.py --input {csv_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Complete WSF Evolution analysis workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Beijing with 50km download radius
  python complete_wsf_workflow.py --city "Beijing, China" --radius 50 --output beijing_results
  
  # Analyze Changzhou with coordinates
  python complete_wsf_workflow.py --lat 31.8122 --lon 119.9692 --radius 25 --output changzhou
  
  # Custom analysis size
  python complete_wsf_workflow.py --city "Shanghai, China" --radius 40 --analysis-size 60 --output shanghai
        """
    )
    
    parser.add_argument('--city', type=str, help='City name')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--radius', type=float, default=25, help='Download radius (km, default: 25)')
    parser.add_argument('--analysis-size', type=float, help='Analysis size (km, default: 2×radius)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./wsf_cache', help='Tile cache directory')
    
    args = parser.parse_args()
    
    # Get coordinates
    if args.city:
        lat, lon = geocode_city(args.city)
        location_name = args.city
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        location_name = f"Location_{lat:.2f}_{lon:.2f}"
    else:
        parser.error("Provide either --city or both --lat and --lon")
    
    # Run workflow
    df = complete_workflow(
        center_lat=lat,
        center_lon=lon,
        radius_km=args.radius,
        output_dir=args.output,
        location_name=location_name,
        cache_dir=args.cache_dir,
        analysis_size_km=args.analysis_size
    )


if __name__ == "__main__":
    main()