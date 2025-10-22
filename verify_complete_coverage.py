#!/usr/bin/env python3
"""
WSF Evolution Complete Coverage Verifier
=========================================

Verifies that you have ALL tiles needed to completely cover a city.

Features:
- Calculates required tiles for a region
- Checks local cache for tiles
- Checks server availability
- Reports coverage completeness
- Identifies missing tiles
- Generates download commands for missing tiles

Usage:
    python verify_complete_coverage.py --city "Beijing, China" --radius 50
    python verify_complete_coverage.py --lat 31.8122 --lon 119.9692 --radius 25
"""

import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from geopy.geocoders import Nominatim
import json


class CompleteCoverageVerifier:
    """Verify complete tile coverage for a city/region"""
    
    BASE_URL = "https://download.geoservice.dlr.de/WSF_EVO/files/"
    TILE_SIZE_DEGREES = 2
    
    def __init__(self, cache_dir: str = "./wsf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_tile_name(self, lat: float, lon: float) -> str:
        """Get tile name with proper signed coordinates"""
        lat_tile = int(np.floor(lat / 2) * 2)
        lon_tile = int(np.floor(lon / 2) * 2)
        return f"WSFevolution_v1_{lon_tile}_{lat_tile}.tif"
    
    def calculate_required_tiles(self, center_lat: float, center_lon: float, 
                                 radius_km: float) -> List[Dict]:
        """Calculate all tiles needed to cover a circular region"""
        # Convert radius to degrees
        lat_radius_deg = radius_km / 111.0
        lon_radius_deg = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        # Calculate bounding box
        min_lat = center_lat - lat_radius_deg
        max_lat = center_lat + lat_radius_deg
        min_lon = center_lon - lon_radius_deg
        max_lon = center_lon + lon_radius_deg
        
        # Calculate tile grid boundaries
        min_lat_tile = int(np.floor(min_lat / 2) * 2)
        max_lat_tile = int(np.floor(max_lat / 2) * 2)
        min_lon_tile = int(np.floor(min_lon / 2) * 2)
        max_lon_tile = int(np.floor(max_lon / 2) * 2)
        
        # Generate all tiles
        tiles = []
        lat_tile = min_lat_tile
        while lat_tile <= max_lat_tile:
            lon_tile = min_lon_tile
            while lon_tile <= max_lon_tile:
                tile_name = f"WSFevolution_v1_{lon_tile}_{lat_tile}.tif"
                
                # Calculate tile bounds
                bounds = {
                    'min_lat': lat_tile,
                    'max_lat': lat_tile + 2,
                    'min_lon': lon_tile,
                    'max_lon': lon_tile + 2,
                }
                
                # Format for display
                lat_hem = 'N' if lat_tile >= 0 else 'S'
                lon_hem = 'E' if lon_tile >= 0 else 'W'
                bounds_str = (f"{abs(lon_tile)}°-{abs(lon_tile+2)}°{lon_hem}, "
                            f"{abs(lat_tile)}°-{abs(lat_tile+2)}°{lat_hem}")
                
                tiles.append({
                    'name': tile_name,
                    'lat_tile': lat_tile,
                    'lon_tile': lon_tile,
                    'bounds': bounds,
                    'bounds_str': bounds_str
                })
                
                lon_tile += 2
            lat_tile += 2
        
        return tiles
    
    def check_local(self, tile_name: str) -> Tuple[bool, float]:
        """Check if tile exists locally"""
        tile_path = self.cache_dir / tile_name
        if tile_path.exists():
            size_mb = tile_path.stat().st_size / (1024 * 1024)
            return True, size_mb
        return False, 0
    
    def check_server(self, tile_name: str) -> Tuple[bool, int, float]:
        """Check if tile exists on server"""
        url = self.BASE_URL + tile_name
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                size_mb = 0
                if 'content-length' in response.headers:
                    size_mb = int(response.headers['content-length']) / (1024 * 1024)
                return True, 200, size_mb
            return False, response.status_code, 0
        except:
            return False, None, 0
    
    def verify_coverage(self, center_lat: float, center_lon: float, 
                       radius_km: float, location_name: str = None) -> Dict:
        """Verify complete coverage for a region"""
        
        print("="*80)
        print("WSF EVOLUTION COMPLETE COVERAGE VERIFICATION")
        print("="*80)
        
        if location_name:
            print(f"\nLocation: {location_name}")
        print(f"Center: ({center_lat:.4f}, {center_lon:.4f})")
        print(f"Radius: {radius_km} km")
        print(f"Cache directory: {self.cache_dir}")
        
        # Calculate required tiles
        required_tiles = self.calculate_required_tiles(center_lat, center_lon, radius_km)
        
        print(f"\n" + "="*80)
        print(f"TILE COVERAGE ANALYSIS")
        print("="*80)
        print(f"\nRequired tiles for complete coverage: {len(required_tiles)}")
        print("\nChecking each tile...")
        print("-"*80)
        
        results = []
        
        for idx, tile_info in enumerate(required_tiles, 1):
            tile_name = tile_info['name']
            print(f"\n[{idx}/{len(required_tiles)}] {tile_name}")
            print(f"  Coverage: {tile_info['bounds_str']}")
            
            # Check local
            local_exists, local_size = self.check_local(tile_name)
            
            if local_exists:
                print(f"  ✓ LOCAL: Found in cache ({local_size:.1f} MB)")
                status = "cached"
                available = True
            else:
                print(f"  ✗ LOCAL: Not in cache")
                
                # Check server
                print(f"  Checking server...", end=" ", flush=True)
                server_exists, status_code, server_size = self.check_server(tile_name)
                
                if server_exists:
                    print(f"✓ AVAILABLE ({server_size:.1f} MB)")
                    status = "available"
                    available = True
                else:
                    print(f"✗ NOT FOUND (HTTP {status_code})")
                    status = "missing"
                    available = False
            
            results.append({
                'tile': tile_name,
                'bounds': tile_info['bounds'],
                'bounds_str': tile_info['bounds_str'],
                'local': local_exists,
                'available': available,
                'status': status
            })
        
        # Summary
        print("\n" + "="*80)
        print("COVERAGE SUMMARY")
        print("="*80)
        
        total = len(results)
        cached = sum(1 for r in results if r['status'] == 'cached')
        available = sum(1 for r in results if r['status'] == 'available')
        missing = sum(1 for r in results if r['status'] == 'missing')
        
        print(f"\nTotal tiles required: {total}")
        print(f"  ✓ Cached locally:   {cached} ({cached/total*100:.0f}%)")
        print(f"  ✓ Available online: {available} ({available/total*100:.0f}%)")
        print(f"  ✗ Missing/unavailable: {missing} ({missing/total*100:.0f}%)")
        
        complete = (cached + available == total)
        
        print("\n" + "-"*80)
        if complete:
            print("✓ COMPLETE COVERAGE AVAILABLE")
            if available > 0:
                print(f"  → {available} tile(s) need to be downloaded")
        else:
            print("✗ INCOMPLETE COVERAGE")
            print(f"  → {missing} tile(s) are not available")
        print("-"*80)
        
        # Missing tiles details
        if missing > 0:
            print("\n" + "="*80)
            print("MISSING TILES (Not Available)")
            print("="*80)
            for r in results:
                if r['status'] == 'missing':
                    print(f"  ✗ {r['tile']}")
                    print(f"    Coverage: {r['bounds_str']}")
            
            print("\n⚠ WARNING: These tiles are not available on the server.")
            print("This means:")
            print("  • The region may not be covered by WSF Evolution dataset")
            print("  • Urban development may be minimal in these areas")
            print("  • You may need to reduce your radius or shift your center")
        
        # Tiles to download
        if available > 0:
            print("\n" + "="*80)
            print("TILES TO DOWNLOAD")
            print("="*80)
            print(f"\nThe following {available} tile(s) need to be downloaded:\n")
            
            for r in results:
                if r['status'] == 'available':
                    print(f"  • {r['tile']}")
                    print(f"    {r['bounds_str']}")
            
            print("\nTo download all missing tiles, run:")
            print(f"  python download_city_tiles.py --lat {center_lat} --lon {center_lon} --radius {radius_km}")
        
        # Ready tiles
        if cached > 0:
            print("\n" + "="*80)
            print("READY FOR ANALYSIS")
            print("="*80)
            print(f"\n{cached} tile(s) already cached and ready:")
            for r in results:
                if r['status'] == 'cached':
                    print(f"  ✓ {r['tile']}")
        
        # Next steps
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        
        if complete and cached == total:
            print("\n✓ All tiles are cached! You can proceed with analysis:")
            print(f"\n  python wsf_workflow_example.py \\")
            print(f"    --lat {center_lat} --lon {center_lon} \\")
            print(f"    --size {radius_km*2} \\")
            print(f"    --output results")
        
        elif complete and available > 0:
            print("\n1. Download the missing tiles:")
            print(f"     python download_city_tiles.py --lat {center_lat} --lon {center_lon} --radius {radius_km}")
            print("\n2. Then run your analysis:")
            print(f"     python wsf_workflow_example.py --lat {center_lat} --lon {center_lon} --size {radius_km*2}")
        
        elif not complete:
            print("\n⚠  Coverage is incomplete. Options:")
            print(f"   1. Reduce radius and try again")
            print(f"   2. Shift center to a better-covered area")
            print(f"   3. Proceed with partial coverage (may affect results)")
            print(f"\nTo find nearby available tiles:")
            print(f"  python check_wsf_tile.py --lat {center_lat} --lon {center_lon} --check-nearby")
        
        return {
            'location': location_name,
            'center': (center_lat, center_lon),
            'radius_km': radius_km,
            'total_tiles': total,
            'cached': cached,
            'available': available,
            'missing': missing,
            'complete': complete,
            'tiles': results
        }


def geocode_city(city_name: str) -> Tuple[float, float]:
    """Geocode city name"""
    print(f"Geocoding: {city_name}")
    geolocator = Nominatim(user_agent="wsf_coverage_verifier")
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            print(f"Found: {location.address}\n")
            return location.latitude, location.longitude
        raise ValueError(f"Could not geocode: {city_name}")
    except Exception as e:
        raise ValueError(f"Geocoding failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify complete WSF Evolution tile coverage for a city',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify coverage for Beijing with 50km radius
  python verify_complete_coverage.py --city "Beijing, China" --radius 50
  
  # Use coordinates directly
  python verify_complete_coverage.py --lat 31.8122 --lon 119.9692 --radius 25
  
  # Save verification report
  python verify_complete_coverage.py --city "Shanghai, China" --radius 40 --save-report
        """
    )
    
    parser.add_argument('--city', type=str, help='City name')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--radius', type=float, default=25, help='Radius in km (default: 25)')
    parser.add_argument('--cache-dir', type=str, default='./wsf_cache', help='Cache directory')
    parser.add_argument('--save-report', action='store_true', help='Save report to JSON')
    
    args = parser.parse_args()
    
    # Get coordinates
    if args.city:
        lat, lon = geocode_city(args.city)
        location_name = args.city
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        location_name = f"({lat:.4f}, {lon:.4f})"
    else:
        parser.error("Provide either --city or both --lat and --lon")
    
    # Verify coverage
    verifier = CompleteCoverageVerifier(cache_dir=args.cache_dir)
    report = verifier.verify_coverage(lat, lon, args.radius, location_name)
    
    # Save report
    if args.save_report:
        report_file = Path("coverage_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved: {report_file}")


if __name__ == "__main__":
    main()