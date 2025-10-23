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
import geopy
from geopy.geocoders import Nominatim
import time
import json
import matplotlib.pyplot as plt
from scipy import ndimage
import rasterio
from rasterio.windows import Window
from rasterio.windows import Window
from rasterio.merge import merge
import pandas as pd
from rasterio.transform import xy

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
    """Analyze built-up areas from WSF Evolution data (single or multiple tiles)"""
    
    # WSF Evolution encoding: each pixel value represents the year when building was detected
    # 0 = no data/no building
    # Values 1985-2015 = year of first detection
    
    def __init__(self):
        self.years = list(range(1985, 2016))
    
    def load_tiles_from_download_result(self, download_result: Dict) -> Tuple[np.ndarray, dict]:
        """
        Load and mosaic tiles from WSFTileManager.download_region() output.
        
        Args:
            download_result: Dictionary returned by WSFTileManager.download_region()
                Must contain 'tiles' list with 'downloaded' and 'path' keys
        
        Returns:
            Tuple of (mosaicked data array, metadata)
        """
        # Extract successfully downloaded tile paths
        tile_paths = []
        for tile_info in download_result['tiles']:
            if tile_info['downloaded'] and tile_info['path']:
                tile_path = Path(tile_info['path'])
                if tile_path.exists():
                    tile_paths.append(tile_path)
        
        if len(tile_paths) == 0:
            raise ValueError("No tiles were successfully downloaded!")
        
        print(f"Loading {len(tile_paths)} tile(s)...")
        
        if len(tile_paths) == 1:
            # Single tile - load directly
            with rasterio.open(tile_paths[0]) as src:
                data = src.read(1)
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': data.shape[1],
                    'height': data.shape[0],
                    'bounds': src.bounds,
                    'num_tiles': 1
                }
                print(f"  Single tile loaded: {data.shape}")
                return data, metadata
        
        else:
            # Multiple tiles - mosaic them
            print("  Mosaicking multiple tiles...")
            src_files = [rasterio.open(path) for path in tile_paths]
            
            try:
                # Merge tiles
                mosaic, out_transform = merge(src_files)
                
                # Calculate bounds from the mosaic transform and shape
                from rasterio.transform import array_bounds
                mosaic_bounds = array_bounds(
                    mosaic.shape[1],  # height
                    mosaic.shape[2],  # width
                    out_transform
                )
                
                # Get first tile for reference metadata
                first_src = src_files[0]
                
                metadata = {
                    'crs': first_src.crs,
                    'transform': out_transform,
                    'width': mosaic.shape[2],
                    'height': mosaic.shape[1],
                    'bounds': mosaic_bounds,
                    'num_tiles': len(tile_paths)
                }
                
                # Extract first band (WSF Evolution is single band)
                data = mosaic[0]
                
                print(f"  Mosaic created: {data.shape}")
                return data, metadata
                
            finally:
                # Close all source files
                for src in src_files:
                    src.close()
    
    def validate_extraction_params(self, data: np.ndarray,
                                   transform: rasterio.Affine,
                                   center_lat: float,
                                   center_lon: float,
                                   size_km: float) -> Tuple[bool, str]:
        """
        Validate if extraction parameters are reasonable.
        
        Returns:
            (is_valid, message)
        """
        from rasterio.transform import rowcol
        
        # Check center point
        row, col = rowcol(transform, center_lon, center_lat)
        
        if row < 0 or row >= data.shape[0] or col < 0 or col >= data.shape[1]:
            return False, f"Center point ({center_lat}, {center_lon}) is outside data bounds"
        
        # Check if requested size is larger than data
        pixels_per_km = 1000 / 30
        size_pixels = int(size_km * pixels_per_km)
        data_size_km = min(data.shape[0], data.shape[1]) * 30 / 1000
        
        if size_km > data_size_km:
            return False, f"Requested size ({size_km}km) is larger than available data (~{data_size_km:.1f}km)"
        
        return True, "OK"
    
    def extract_built_area_bbox(self, data: np.ndarray, 
                                transform: rasterio.Affine,
                                center_lat: float, 
                                center_lon: float,
                                size_km: float = 10) -> Tuple[np.ndarray, dict]:
        """
        Extract a bounding box around a location from mosaicked data.
        
        Args:
            data: Mosaicked WSF Evolution array
            transform: Affine transform from the mosaic
            center_lat: Latitude of center point
            center_lon: Longitude of center point
            size_km: Size of bounding box in kilometers (default 10km = 10km x 10km)
        
        Returns:
            Array with WSF data and metadata dictionary
        """
        # Convert lat/lon to pixel coordinates using the transform
        from rasterio.transform import rowcol
        row, col = rowcol(transform, center_lon, center_lat)
        
        # Check if center point is within data bounds
        if row < 0 or row >= data.shape[0] or col < 0 or col >= data.shape[1]:
            print(f"\n⚠️  Warning: Center point ({center_lat}, {center_lon}) is outside data bounds!")
            print(f"  Center pixel: ({row}, {col})")
            print(f"  Data shape: {data.shape}")
            print(f"  Adjusting to use full available data...")
            
            # Use the full data instead
            return data, {
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km,
                'adjusted': True
            }
        
        # Calculate pixel extent (approximately)
        # At 30m resolution: 1km ≈ 33.33 pixels
        pixels_per_km = 1000 / 30  # 30m pixel size
        half_size_pixels = int((size_km / 2) * pixels_per_km)
        
        # Define extraction bounds with clipping
        row_start = max(0, row - half_size_pixels)
        row_end = min(data.shape[0], row + half_size_pixels)
        col_start = max(0, col - half_size_pixels)
        col_end = min(data.shape[1], col + half_size_pixels)
        
        # Validate bounds
        if row_start >= row_end or col_start >= col_end:
            print(f"\n⚠️  Warning: Invalid extraction bounds!")
            print(f"  Requested size: {size_km} km × {size_km} km")
            print(f"  Row range: {row_start} to {row_end} (height: {row_end - row_start})")
            print(f"  Col range: {col_start} to {col_end} (width: {col_end - col_start})")
            print(f"  Using full data instead...")
            
            return data, {
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km,
                'adjusted': True
            }
        
        # Extract subset
        subset = data[row_start:row_end, col_start:col_end]
        
        # Additional validation
        if subset.shape[0] == 0 or subset.shape[1] == 0:
            print(f"\n⚠️  Warning: Extracted subset is empty!")
            print(f"  Using full data instead...")
            
            return data, {
                'transform': transform,
                'width': data.shape[1],
                'height': data.shape[0],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'size_km': size_km,
                'adjusted': True
            }
        
        # Calculate new transform for the subset
        from rasterio.windows import transform as window_transform, Window
        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        subset_transform = window_transform(window, transform)
        
        metadata = {
            'transform': subset_transform,
            'width': subset.shape[1],
            'height': subset.shape[0],
            'center_lat': center_lat,
            'center_lon': center_lon,
            'size_km': size_km,
            'adjusted': False
        }
        
        # Report if size was adjusted due to clipping
        actual_size_km = min(
            (row_end - row_start) * 30 / 1000,
            (col_end - col_start) * 30 / 1000
        )
        
        if abs(actual_size_km - size_km) > 1:  # More than 1km difference
            print(f"\n  Note: Requested {size_km}km × {size_km}km, got ~{actual_size_km:.1f}km × {actual_size_km:.1f}km")
            print(f"  (clipped to data bounds)")
        else:
            print(f"Extracted {size_km}km × {size_km}km region: {subset.shape}")
        
        return subset, metadata
    
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
    
    def calculate_perimeter(self, binary_mask: np.ndarray) -> Tuple[int, float]:
        """
        Calculate perimeter of a binary mask.
        
        Args:
            binary_mask: Binary array where 1 = built, 0 = not built
        
        Returns:
            Tuple of (perimeter in pixels, perimeter in km)
        """
        # Erode by 1 pixel
        eroded = ndimage.binary_erosion(binary_mask)
        
        # Perimeter = original - eroded
        perimeter_mask = binary_mask - eroded
        perimeter_pixels = int(perimeter_mask.sum())
        
        # Convert to km (30m resolution)
        perimeter_km = perimeter_pixels * 0.03  # 30m = 0.03km
        
        return perimeter_pixels, perimeter_km
    
    def analyze_evolution(self, wsf_data: np.ndarray) -> Dict:
        """
        Analyze urban evolution across all years.
        
        Returns:
            Dictionary with year-by-year statistics including perimeter
        """
        results = {
            'years': [],
            'total_built_pixels': [],
            'lcc_pixels': [],
            'lcc_percentage': [],
            'num_components': [],
            'lcc_area_km2': [],
            'perimeter_pixels': [],
            'perimeter_km': []
        }
        
        pixel_area_km2 = (30 * 30) / 1e6  # 30m x 30m in km²
        
        for year in self.years:
            mask = self.extract_year_mask(wsf_data, year)
            total_built = mask.sum()
            
            lcc_mask, lcc_size = self.find_largest_connected_component(mask)
            
            # Calculate perimeter
            perim_pixels, perim_km = self.calculate_perimeter(lcc_mask)
            
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
            results['perimeter_pixels'].append(perim_pixels)
            results['perimeter_km'].append(round(perim_km, 2))
            
            print(f"{year}: LCC={lcc_size:,} px ({lcc_area:.2f} km²), "
                  f"Perimeter={perim_pixels:,} px ({perim_km:.1f} km), "
                  f"Components={num_components}")
        
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
            lcc_mask, lcc_size = self.find_largest_connected_component(mask)
            
            # Calculate perimeter
            perim_pixels, perim_km = self.calculate_perimeter(lcc_mask)
            
            # Create RGB visualization
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[mask == 1] = [200, 200, 200]  # Gray for all built areas
            rgb[lcc_mask == 1] = [255, 0, 0]  # Red for LCC
            
            axes[idx].imshow(rgb)
            axes[idx].set_title(f'{year}\nLCC: {lcc_size:,} px\nPerim: {perim_km:.1f} km')
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

def analyze_from_download_result(download_result: Dict, 
                                  center_lat: float = None,
                                  center_lon: float = None,
                                  size_km: float = None,
                                  output_path: str = None) -> Dict:
    """
    Convenience function to analyze tiles from download_region() result.
    
    Args:
        download_result: Output from WSFTileManager.download_region()
        center_lat: Optional center latitude to extract subset
        center_lon: Optional center longitude to extract subset
        size_km: Optional size for subset extraction (km)
        output_path: Optional path to save visualization
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = BuiltAreaAnalyzer()
    
    # Load and mosaic tiles
    print("="*70)
    print("LOADING TILES")
    print("="*70)
    data, metadata = analyzer.load_tiles_from_download_result(download_result)
    
    # Extract subset if requested
    if center_lat is not None and center_lon is not None and size_km is not None:
        print("\n" + "="*70)
        print("EXTRACTING REGION OF INTEREST")
        print("="*70)
        data, metadata = analyzer.extract_built_area_bbox(
            data, metadata['transform'], center_lat, center_lon, size_km
        )
    
    # Analyze evolution
    print("\n" + "="*70)
    print("ANALYZING URBAN EVOLUTION")
    print("="*70)
    results = analyzer.analyze_evolution(data)
    
    # Create visualization
    if output_path:
        print("\n" + "="*70)
        print("CREATING VISUALIZATION")
        print("="*70)
        analyzer.visualize_evolution(data, output_path)
    
    return results



def analyze_from_download_result(download_result: Dict, 
                                  center_lat: float = None,
                                  center_lon: float = None,
                                  size_km: float = None,
                                  output_path: str = None) -> Dict:
    """
    Convenience function to analyze tiles from download_region() result.
    
    Args:
        download_result: Output from WSFTileManager.download_region()
        center_lat: Optional center latitude to extract subset
        center_lon: Optional center longitude to extract subset
        size_km: Optional size for subset extraction (km)
        output_path: Optional path to save visualization
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = BuiltAreaAnalyzer()
    
    # Load and mosaic tiles
    print("="*70)
    print("LOADING TILES")
    print("="*70)
    data, metadata = analyzer.load_tiles_from_download_result(download_result)
    
    # Extract subset if requested
    if center_lat is not None and center_lon is not None and size_km is not None:
        print("\n" + "="*70)
        print("EXTRACTING REGION OF INTEREST")
        print("="*70)
        data, metadata = analyzer.extract_built_area_bbox(
            data, metadata['transform'], center_lat, center_lon, size_km
        )
    
    # Analyze evolution
    print("\n" + "="*70)
    print("ANALYZING URBAN EVOLUTION")
    print("="*70)
    results = analyzer.analyze_evolution(data)
    
    # Create visualization
    if output_path:
        print("\n" + "="*70)
        print("CREATING VISUALIZATION")
        print("="*70)
        analyzer.visualize_evolution(data, output_path)
    
    return results


def mask_to_coordinates(mask: np.ndarray, transform: rasterio.Affine) -> pd.DataFrame:
    """
    Convert binary mask to dataframe with pixel coordinates.
    
    Args:
        mask: Binary mask (1 = LCC, 0 = not LCC)
        transform: Affine transform for geographic coordinates
    
    Returns:
        DataFrame with columns: row, col, latitude, longitude
    """
    # Get pixel indices where mask is 1
    rows, cols = np.where(mask == 1)
    
    # Convert pixel coordinates to geographic coordinates
    lats = []
    lons = []
    
    for row, col in zip(rows, cols):
        # Get center of pixel in geographic coordinates
        lon, lat = xy(transform, row, col, offset='center')
        lats.append(lat)
        lons.append(lon)
    
    return pd.DataFrame({
        'row': rows,
        'col': cols,
        'latitude': lats,
        'longitude': lons
    })


def export_lcc_coordinates_all_years(wsf_data: np.ndarray,
                                     transform: rasterio.Affine,
                                     analyzer: BuiltAreaAnalyzer,
                                     output_path: str,
                                     years: List[int] = None) -> str:
    """
    Export LCC coordinates for all years to CSV.
    
    Args:
        wsf_data: WSF Evolution array
        transform: Affine transform
        analyzer: BuiltAreaAnalyzer instance
        output_path: Output CSV file path
        years: List of years to export (default: 1985-2015)
    
    Returns:
        Path to saved CSV file
    """
    if years is None:
        years = list(range(1985, 2016))
    
    all_data = []
    
    print("\nExtracting LCC coordinates for each year...")
    print("-" * 70)
    
    for idx, year in enumerate(years, 1):
        print(f"Processing {year} ({idx}/{len(years)})...", end='\r')
        
        # Extract mask for this year
        mask = analyzer.extract_year_mask(wsf_data, year)
        
        # Get LCC
        lcc_mask, lcc_size = analyzer.find_largest_connected_component(mask)
        
        if lcc_size == 0:
            print(f"  Year {year}: No LCC found, skipping")
            continue
        
        # Convert mask to coordinates
        coords_df = mask_to_coordinates(lcc_mask, transform)
        
        # Add year column
        coords_df['year'] = year
        
        # Add to list
        all_data.append(coords_df)
        
        print(f"  Year {year}: {lcc_size:,} pixels extracted", end='\r')
    
    print("\n" + "-" * 70)
    
    # Combine all years
    print("\nCombining data from all years...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: year, latitude, longitude, row, col
    combined_df = combined_df[['year', 'latitude', 'longitude', 'row', 'col']]
    
    # Save to CSV
    output_path = Path(output_path)
    combined_df.to_csv(output_path, index=False)
    
    print(f"✓ Saved: {output_path}")
    print(f"  Total pixels: {len(combined_df):,}")
    print(f"  Years: {combined_df['year'].min()}-{combined_df['year'].max()}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return str(output_path)





# Example usage
if __name__ == "__main__":
    """
    Example: How to use with WSFTileManager
    """
    
    # Example 1: Direct usage with download result
    print("""
EXAMPLE USAGE:
==============

# Step 1: Download tiles
from download_city_tiles import WSFTileManager

manager = WSFTileManager(cache_dir="./wsf_cache")
download_result = manager.download_region(
    center_lat=31.8122,
    center_lon=119.9692,
    radius_km=25
)

# Step 2: Analyze the downloaded tiles
from built_area_analyzer import analyze_from_download_result

results = analyze_from_download_result(
    download_result=download_result,
    center_lat=31.8122,
    center_lon=119.9692,
    size_km=50,  # Extract 50km x 50km region
    output_path="urban_evolution.png"
)

# Step 3: Use the results
import pandas as pd
df = pd.DataFrame(results)
print(df)

# Access specific metrics
print(f"2015 LCC area: {results['lcc_area_km2'][-1]} km²")
print(f"2015 Perimeter: {results['perimeter_km'][-1]} km")
    """)
    
    # Example 2: Manual step-by-step
    print("""
MANUAL STEP-BY-STEP:
====================

# Step 1: Download tiles
from download_city_tiles import WSFTileManager

manager = WSFTileManager(cache_dir="./wsf_cache")
download_result = manager.download_region(
    center_lat=31.8122,
    center_lon=119.9692,
    radius_km=25
)

# Step 2: Create analyzer
from built_area_analyzer import BuiltAreaAnalyzer

analyzer = BuiltAreaAnalyzer()

# Step 3: Load tiles
data, metadata = analyzer.load_tiles_from_download_result(download_result)

# Step 4: Extract region (optional)
data_subset, meta_subset = analyzer.extract_built_area_bbox(
    data=data,
    transform=metadata['transform'],
    center_lat=31.8122,
    center_lon=119.9692,
    size_km=50
)

# Step 5: Analyze
results = analyzer.analyze_evolution(data_subset)

# Step 6: Visualize
analyzer.visualize_evolution(data_subset, "evolution.png")

# Step 7: Create DataFrame
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("timeseries.csv", index=False)
    """)
















