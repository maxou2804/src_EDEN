#!/usr/bin/env python3
"""
WSF Evolution Multi-Tile Downloader
====================================

This script identifies and downloads ALL tiles needed to cover a city or region.
Since tiles are 2°×2°, a large city might span multiple tiles.

Usage:
    python download_city_tiles.py --city "Beijing, China" --radius 50
    python download_city_tiles.py --lat 31.8122 --lon 119.9692 --radius 30
"""

import numpy as np
import requests
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from geopy.geocoders import Nominatim
import time
import json
import matplotlib.pyplot as plt
from scipy import ndimage
import rasterio
from rasterio.windows import Window

class WSFTileManager:
    """Manage downloading multiple WSF Evolution tiles for a region"""
    
    BASE_URL = "https://download.geoservice.dlr.de/WSF_EVO/files/"
    TILE_SIZE_DEGREES = 2  # Tiles are 2° × 2°
    
    def __init__(self, cache_dir: str = "./wsf_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_tile_name(self, lat: float, lon: float) -> str:
        """Get tile name for a specific lat/lon"""
        lat_tile = int(np.floor(lat / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        lon_tile = int(np.floor(lon / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        return f"WSFevolution_v1_{lon_tile}_{lat_tile}.tif"
    
    def get_tile_bounds(self, tile_name: str) -> Tuple[float, float, float, float]:
        """Get the bounds of a tile from its name"""
        # Parse: WSFevolution_v1_{lon}_{lat}.tif
        parts = tile_name.replace('.tif', '').split('_')
        lon_tile = int(parts[2])
        lat_tile = int(parts[3])
        
        return (
            lat_tile,  # min_lat
            lat_tile + self.TILE_SIZE_DEGREES,  # max_lat
            lon_tile,  # min_lon
            lon_tile + self.TILE_SIZE_DEGREES   # max_lon
        )
    
    def calculate_required_tiles(self, center_lat: float, center_lon: float, 
                                 radius_km: float) -> List[Tuple[str, dict]]:
        """
        Calculate all tiles needed to cover a circular region.
        
        Args:
            center_lat, center_lon: Center coordinates
            radius_km: Radius in kilometers
        
        Returns:
            List of (tile_name, tile_info_dict) tuples
        """
        # Convert radius to degrees (approximate)
        # 1 degree ≈ 111 km at equator
        # Longitude degrees vary with latitude
        lat_radius_deg = radius_km / 111.0
        lon_radius_deg = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        # Calculate bounding box
        min_lat = center_lat - lat_radius_deg
        max_lat = center_lat + lat_radius_deg
        min_lon = center_lon - lon_radius_deg
        max_lon = center_lon + lon_radius_deg
        
        # Find all tiles that intersect with this bounding box
        tiles = []
        
        # Calculate tile grid boundaries
        min_lat_tile = int(np.floor(min_lat / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        max_lat_tile = int(np.floor(max_lat / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        min_lon_tile = int(np.floor(min_lon / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        max_lon_tile = int(np.floor(max_lon / self.TILE_SIZE_DEGREES) * self.TILE_SIZE_DEGREES)
        
        # Iterate through all tiles in the grid
        lat_tile = min_lat_tile
        while lat_tile <= max_lat_tile:
            lon_tile = min_lon_tile
            while lon_tile <= max_lon_tile:
                tile_name = f"WSFevolution_v1_{lon_tile}_{lat_tile}.tif"
                
                tile_info = {
                    'lat_tile': lat_tile,
                    'lon_tile': lon_tile,
                    'bounds': (lat_tile, lat_tile + self.TILE_SIZE_DEGREES,
                              lon_tile, lon_tile + self.TILE_SIZE_DEGREES),
                    'center': (lat_tile + 1, lon_tile + 1)
                }
                
                tiles.append((tile_name, tile_info))
                
                lon_tile += self.TILE_SIZE_DEGREES
            lat_tile += self.TILE_SIZE_DEGREES
        
        return tiles
    
    def check_tile_exists(self, tile_name: str) -> Tuple[bool, int]:
        """Check if a tile exists on the server"""
        url = self.BASE_URL + tile_name
        
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200, response.status_code
        except requests.exceptions.RequestException:
            return False, None
    
    def download_tile(self, tile_name: str, force_redownload: bool = False) -> Tuple[bool, Path]:
        """
        Download a tile if not already cached.
        
        Returns:
            (success: bool, path: Path)
        """
        tile_path = self.cache_dir / tile_name
        
        # Check if already cached
        if tile_path.exists() and not force_redownload:
            print(f"  ✓ Using cached: {tile_name}")
            return True, tile_path
        
        url = self.BASE_URL + tile_name
        
        try:
            print(f"  ⬇ Downloading: {tile_name}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(tile_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"    Progress: {percent:.1f}%", end='\r', flush=True)
            
            print(f"\n  ✓ Downloaded: {tile_name} ({total_size/(1024*1024):.1f} MB)")
            return True, tile_path
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Failed to download {tile_name}: {e}")
            if tile_path.exists():
                tile_path.unlink()
            return False, None
    
    def download_region(self, center_lat: float, center_lon: float, 
                       radius_km: float) -> Dict:
        """
        Download all tiles needed to cover a region.
        
        Returns:
            Dictionary with download results
        """
        print("="*70)
        print("WSF EVOLUTION MULTI-TILE DOWNLOADER")
        print("="*70)
        print(f"\nCenter: ({center_lat:.4f}, {center_lon:.4f})")
        print(f"Radius: {radius_km} km")
        
        # Calculate required tiles
        required_tiles = self.calculate_required_tiles(center_lat, center_lon, radius_km)
        
        print(f"\nRequired tiles: {len(required_tiles)}")
        print("-"*70)
        
        results = {
            'center': (center_lat, center_lon),
            'radius_km': radius_km,
            'required_tiles': len(required_tiles),
            'tiles': []
        }
        
        # Check and download each tile
        for idx, (tile_name, tile_info) in enumerate(required_tiles, 1):
            print(f"\n[{idx}/{len(required_tiles)}] {tile_name}")
            print(f"  Bounds: [{tile_info['bounds'][2]}°E to {tile_info['bounds'][3]}°E, "
                  f"{tile_info['bounds'][0]}°N to {tile_info['bounds'][1]}°N]")
            
            # Check if tile exists on server
            exists, status_code = self.check_tile_exists(tile_name)
            
            tile_result = {
                'name': tile_name,
                'bounds': tile_info['bounds'],
                'exists_on_server': exists,
                'status_code': status_code,
                'downloaded': False,
                'path': None
            }
            
            if exists:
                # Download tile
                success, path = self.download_tile(tile_name)
                tile_result['downloaded'] = success
                tile_result['path'] = str(path) if path else None
            else:
                print(f"  ✗ Not available on server (Status: {status_code})")
            
            results['tiles'].append(tile_result)
        
        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        
        available = sum(1 for t in results['tiles'] if t['exists_on_server'])
        downloaded = sum(1 for t in results['tiles'] if t['downloaded'])
        
        print(f"Total tiles required: {len(required_tiles)}")
        print(f"Available on server: {available}")
        print(f"Successfully downloaded: {downloaded}")
        print(f"Already cached: {available - downloaded}")
        print(f"Not available: {len(required_tiles) - available}")
        
        if available < len(required_tiles):
            print("\n⚠ WARNING: Some tiles are not available!")
            print("This may result in incomplete coverage of your region.")
            print("Missing tiles:")
            for tile in results['tiles']:
                if not tile['exists_on_server']:
                    print(f"  - {tile['name']}")
        
        return results
    
    def visualize_coverage(self, results: Dict, output_file: str = None):
        """Create a visualization of tile coverage"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        center_lat, center_lon = results['center']
        radius_km = results['radius_km']
        
        # Draw circle representing the desired coverage
        lat_radius = radius_km / 111.0
        lon_radius = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        circle = plt.Circle((center_lon, center_lat), 
                           max(lat_radius, lon_radius),
                           fill=False, edgecolor='red', linewidth=2,
                           linestyle='--', label='Requested coverage')
        ax.add_patch(circle)
        
        # Draw tiles
        for tile in results['tiles']:
            bounds = tile['bounds']
            min_lat, max_lat, min_lon, max_lon = bounds
            
            if tile['downloaded']:
                color = 'green'
                alpha = 0.3
                label = 'Downloaded'
            elif tile['exists_on_server']:
                color = 'yellow'
                alpha = 0.3
                label = 'Available'
            else:
                color = 'red'
                alpha = 0.2
                label = 'Not available'
            
            rect = patches.Rectangle((min_lon, min_lat), 
                                     max_lon - min_lon,
                                     max_lat - min_lat,
                                     linewidth=1, edgecolor='black',
                                     facecolor=color, alpha=alpha)
            ax.add_patch(rect)
            
            # Add tile name
            ax.text((min_lon + max_lon) / 2, (min_lat + max_lat) / 2,
                   tile['name'].replace('WSFevolution_v1_', '').replace('.tif', ''),
                   ha='center', va='center', fontsize=8)
        
        # Mark center
        ax.plot(center_lon, center_lat, 'r*', markersize=15, label='Center')
        
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'WSF Evolution Tile Coverage\nCenter: ({center_lat:.4f}, {center_lon:.4f}), Radius: {radius_km}km')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.3, label='Downloaded'),
            Patch(facecolor='yellow', alpha=0.3, label='Available (not downloaded)'),
            Patch(facecolor='red', alpha=0.2, label='Not available'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Requested coverage'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='r', 
                      markersize=10, label='Center'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n✓ Coverage map saved: {output_file}")
        else:
            plt.show()


def geocode_city(city_name: str) -> Tuple[float, float]:
    """Geocode city name to coordinates"""
    print(f"Geocoding: {city_name}")
    geolocator = Nominatim(user_agent="wsf_tile_downloader")
    
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            print(f"Found: {location.address}")
            return location.latitude, location.longitude
        else:
            raise ValueError(f"Could not geocode: {city_name}")
    except Exception as e:
        raise ValueError(f"Geocoding failed: {e}")


class BuiltAreaAnalyzer:
    """Analyze built-up areas from WSF Evolution data"""
    
    # WSF Evolution encoding: each pixel value represents the year when building was detected
    # 0 = no data/no building
    # Values 1985-2015 = year of first detection
    
    def __init__(self):
        self.years = list(range(1985, 2016))
    
    def extract_built_area_bbox(self, raster_path: Path, 
                                center_lat: float, center_lon: float,
                                size_km: float = 10) -> Tuple[np.ndarray, dict]:
        """
        Extract a bounding box around a location from the raster.
        
        Args:
            raster_path: Path to the WSF Evolution GeoTIFF
            center_lat: Latitude of center point
            center_lon: Longitude of center point
            size_km: Size of bounding box in kilometers (default 10km = 10km x 10km)
        
        Returns:
            Array with WSF data and metadata dictionary
        """
        with rasterio.open(raster_path) as src:
            # Convert center point to pixel coordinates
            row, col = src.index(center_lon, center_lat)
            
            # Calculate pixel extent (approximately)
            # At 30m resolution: 1km ≈ 33.33 pixels
            pixels_per_km = 1000 / 30  # 30m pixel size
            half_size_pixels = int((size_km / 2) * pixels_per_km)
            
            # Define window
            window = Window(
                col - half_size_pixels,
                row - half_size_pixels,
                half_size_pixels * 2,
                half_size_pixels * 2
            )
            
            # Read data
            data = src.read(1, window=window)
            
            # Get transform for the window
            transform = src.window_transform(window)
            
            metadata = {
                'crs': src.crs,
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km
            }
            
            return data, metadata
    
    def extract_year_mask(self, wsf_data: np.ndarray, year: int) -> np.ndarray:
        """
        Extract binary mask of built areas up to and including a specific year.
        
        Args:
            wsf_data: WSF Evolution array (values are years 1985-2015 or 0 for no building)
            year: Target year
        
        Returns:
            Binary mask where 1 = built by this year, 0 = not built
        """
        # Buildings detected in years <= target year
        mask = ((wsf_data > 0) & (wsf_data <= year)).astype(np.uint8)
        return mask
    
    def find_largest_connected_component(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Find the largest connected component in a binary mask.
        
        Args:
            binary_mask: Binary array where 1 = built, 0 = not built
        
        Returns:
            Tuple of (LCC mask, LCC size in pixels)
        """
        if binary_mask.sum() == 0:
            return binary_mask, 0
        
        # Label connected components (8-connectivity)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            return np.zeros_like(binary_mask), 0
        
        # Find largest component
        component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background
        largest_label = component_sizes.argmax() + 1
        
        lcc_mask = (labeled_array == largest_label).astype(np.uint8)
        lcc_size = component_sizes[largest_label - 1]
        
        return lcc_mask, int(lcc_size)
    
    def analyze_evolution(self, wsf_data: np.ndarray) -> Dict:
        """
        Analyze urban evolution across all years.
        
        Returns:
            Dictionary with year-by-year statistics
        """
        results = {
            'years': [],
            'total_built_pixels': [],
            'lcc_pixels': [],
            'lcc_percentage': [],
            'num_components': [],
            'lcc_area_km2': []
        }
        
        pixel_area_km2 = (30 * 30) / 1e6  # 30m x 30m in km²
        
        for year in self.years:
            mask = self.extract_year_mask(wsf_data, year)
            total_built = mask.sum()
            
            lcc_mask, lcc_size = self.find_largest_connected_component(mask)
            
            # Count total number of components
            labeled_array, num_components = ndimage.label(mask)
            
            lcc_pct = (lcc_size / total_built * 100) if total_built > 0 else 0
            lcc_area = lcc_size * pixel_area_km2
            
            results['years'].append(year)
            results['total_built_pixels'].append(int(total_built))
            results['lcc_pixels'].append(lcc_size)
            results['lcc_percentage'].append(round(lcc_pct, 2))
            results['num_components'].append(num_components)
            results['lcc_area_km2'].append(round(lcc_area, 3))
            
            print(f"{year}: Total={total_built:,} px, LCC={lcc_size:,} px "
                  f"({lcc_pct:.1f}%), Components={num_components}, "
                  f"LCC Area={lcc_area:.3f} km²")
        
        return results
    
    def visualize_evolution(self, wsf_data: np.ndarray, output_path: str = None):
        """Create visualization of urban evolution"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Select representative years
        selected_years = [1985, 1990, 1995, 2000, 2005, 2010, 2015]
        
        for idx, year in enumerate(selected_years):
            if idx >= len(axes):
                break
                
            mask = self.extract_year_mask(wsf_data, year)
            lcc_mask, _ = self.find_largest_connected_component(mask)
            
            # Create RGB visualization
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[mask == 1] = [200, 200, 200]  # Gray for all built areas
            rgb[lcc_mask == 1] = [255, 0, 0]  # Red for LCC
            
            axes[idx].imshow(rgb)
            axes[idx].set_title(f'{year}')
            axes[idx].axis('off')
        
        # Hide extra subplot
        if len(selected_years) < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle('Urban Evolution: Gray=All Built Areas, Red=Largest Connected Component')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()





















