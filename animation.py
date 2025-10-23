#!/usr/bin/env python3
"""
WSF Evolution LCC Animation Tool
=================================

Creates animated visualizations showing year-by-year urban growth.

Supports:
- GIF animations
- MP4 videos
- Frame-by-frame PNG sequences

Usage:
    python create_lcc_animation.py --lat 31.8122 --lon 119.9692 --radius 25 --output changzhou_evolution.gif
    python create_lcc_animation.py --city "Beijing, China" --radius 50 --output beijing_evolution.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))
from wsf_evolution_lcc import WSFTileManager
from wsf_evolution_lcc import BuiltAreaAnalyzer
from geopy.geocoders import Nominatim


class LCCAnimator:
    """Create animations of LCC evolution"""
    
    def __init__(self):
        self.years = list(range(1985, 2016))
        self.cmap_built = plt.cm.Greys  # Colormap for built areas
        self.cmap_lcc = plt.cm.Reds     # Colormap for LCC
    
    def create_animation_frames(self, wsf_data: np.ndarray, 
                               analyzer: BuiltAreaAnalyzer) -> List[Dict]:
        """
        Create all animation frames with metrics.
        
        Returns:
            List of dicts with: year, mask, lcc_mask, metrics
        """
        frames = []
        
        print("\nGenerating animation frames...")
        for idx, year in enumerate(self.years, 1):
            # Extract masks
            mask = analyzer.extract_year_mask(wsf_data, year)
            lcc_mask, lcc_size = analyzer.find_largest_connected_component(mask)
            
            # Calculate metrics
            perim_px, perim_km = analyzer.calculate_perimeter(lcc_mask)
            pixel_area_km2 = (30 * 30) / 1e6
            lcc_area_km2 = lcc_size * pixel_area_km2
            
            from scipy import ndimage
            _, num_components = ndimage.label(mask)
            
            total_built = mask.sum()
            lcc_pct = (lcc_size / total_built * 100) if total_built > 0 else 0
            
            frames.append({
                'year': year,
                'mask': mask,
                'lcc_mask': lcc_mask,
                'lcc_size': lcc_size,
                'lcc_area_km2': lcc_area_km2,
                'perimeter_pixels': perim_px,
                'perimeter_km': perim_km,
                'num_components': num_components,
                'lcc_percentage': lcc_pct,
                'total_built': total_built
            })
            
            print(f"  Frame {idx}/{len(self.years)}: {year} - "
                  f"LCC={lcc_size:,} px, Perim={perim_px:,} px", end='\r')
        
        print("\n✓ All frames generated")
        return frames
    
    def create_gif_animation(self, frames: List[Dict], output_path: str,
                            fps: int = 2, show_metrics: bool = True,
                            show_growth: bool = True) -> str:
        """
        Create GIF animation.
        
        Args:
            frames: List of frame dictionaries
            output_path: Output file path (.gif)
            fps: Frames per second (default: 2)
            show_metrics: Show metrics overlay
            show_growth: Highlight new growth
        
        Returns:
            Path to saved file
        """
        print(f"\nCreating GIF animation...")
        print(f"  Frames: {len(frames)}")
        print(f"  FPS: {fps}")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Storage for previous frame (for growth highlighting)
        previous_lcc = None
        
        def update_frame(frame_idx):
            nonlocal previous_lcc
            ax.clear()
            
            frame = frames[frame_idx]
            year = frame['year']
            
            # Create RGB image
            rgb = np.zeros((*frame['mask'].shape, 3), dtype=np.uint8)
            
            if show_growth and previous_lcc is not None:
                # Show new growth in green
                new_growth = (frame['lcc_mask'] == 1) & (previous_lcc == 0)
                rgb[new_growth] = [0, 255, 0]  # Green for new growth
                
                # Old built areas in gray
                old_built = (previous_lcc == 1)
                rgb[old_built] = [200, 200, 200]  # Gray for old
                
                # Current LCC boundary in red
                rgb[frame['lcc_mask'] == 1] = [255, 100, 100]  # Red for current LCC
            else:
                # Standard visualization
                rgb[frame['mask'] == 1] = [220, 220, 220]  # Light gray for all built
                rgb[frame['lcc_mask'] == 1] = [255, 50, 50]  # Red for LCC
            
            # Display image
            ax.imshow(rgb)
            ax.axis('off')
            
            # Title with year
            title = f'Year: {year}'
            if show_metrics:
                title += f'\nLCC Area: {frame["lcc_area_km2"]:.2f} km²  |  '
                title += f'Perimeter: {frame["perimeter_km"]:.1f} km  |  '
                title += f'Components: {frame["num_components"]}'
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Add metrics box
            if show_metrics:
                metrics_text = (
                    f'LCC: {frame["lcc_size"]:,} pixels\n'
                    f'Perim: {frame["perimeter_pixels"]:,} pixels\n'
                    f'Coverage: {frame["lcc_percentage"]:.1f}%'
                )
                
                # Add text box
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=props,
                       family='monospace')
            
            # Add legend
            if show_growth and previous_lcc is not None:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='gray', label='Previous LCC'),
                    Patch(facecolor='red', label='Current LCC'),
                    Patch(facecolor='green', label='New Growth')
                ]
                ax.legend(handles=legend_elements, loc='upper right', 
                         framealpha=0.8, fontsize=10)
            else:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='lightgray', label='All Built Areas'),
                    Patch(facecolor='red', label='Largest Component')
                ]
                ax.legend(handles=legend_elements, loc='upper right',
                         framealpha=0.8, fontsize=10)
            
            # Update previous frame
            previous_lcc = frame['lcc_mask'].copy()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(frames),
            interval=1000/fps, repeat=True
        )
        
        # Save as GIF
        output_path = Path(output_path)
        if output_path.suffix != '.gif':
            output_path = output_path.with_suffix('.gif')
        
        print(f"  Saving to: {output_path}")
        anim.save(str(output_path), writer='pillow', fps=fps, dpi=100)
        plt.close()
        
        print(f"✓ GIF animation saved: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        return str(output_path)
    
    def create_video_animation(self, frames: List[Dict], output_path: str,
                               fps: int = 5, show_metrics: bool = True,
                               show_growth: bool = True, codec: str = 'libx264') -> str:
        """
        Create MP4 video animation.
        
        Args:
            frames: List of frame dictionaries
            output_path: Output file path (.mp4)
            fps: Frames per second (default: 5)
            show_metrics: Show metrics overlay
            show_growth: Highlight new growth
            codec: Video codec (default: libx264)
        
        Returns:
            Path to saved file
        """
        print(f"\nCreating MP4 video animation...")
        print(f"  Frames: {len(frames)}")
        print(f"  FPS: {fps}")
        print(f"  Codec: {codec}")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        previous_lcc = None
        
        def update_frame(frame_idx):
            nonlocal previous_lcc
            ax.clear()
            
            frame = frames[frame_idx]
            year = frame['year']
            
            # Create RGB image (same as GIF)
            rgb = np.zeros((*frame['mask'].shape, 3), dtype=np.uint8)
            
            if show_growth and previous_lcc is not None:
                new_growth = (frame['lcc_mask'] == 1) & (previous_lcc == 0)
                rgb[new_growth] = [0, 255, 0]
                old_built = (previous_lcc == 1)
                rgb[old_built] = [200, 200, 200]
                rgb[frame['lcc_mask'] == 1] = [255, 100, 100]
            else:
                rgb[frame['mask'] == 1] = [220, 220, 220]
                rgb[frame['lcc_mask'] == 1] = [255, 50, 50]
            
            ax.imshow(rgb)
            ax.axis('off')
            
            # Title and metrics
            title = f'Year: {year}'
            if show_metrics:
                title += f'\nLCC Area: {frame["lcc_area_km2"]:.2f} km²  |  '
                title += f'Perimeter: {frame["perimeter_km"]:.1f} km  |  '
                title += f'Components: {frame["num_components"]}'
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            if show_metrics:
                metrics_text = (
                    f'LCC: {frame["lcc_size"]:,} pixels\n'
                    f'Perim: {frame["perimeter_pixels"]:,} pixels\n'
                    f'Coverage: {frame["lcc_percentage"]:.1f}%'
                )
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=props,
                       family='monospace')
            
            if show_growth and previous_lcc is not None:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='gray', label='Previous LCC'),
                    Patch(facecolor='red', label='Current LCC'),
                    Patch(facecolor='green', label='New Growth')
                ]
                ax.legend(handles=legend_elements, loc='upper right',
                         framealpha=0.8, fontsize=10)
            
            previous_lcc = frame['lcc_mask'].copy()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(frames),
            interval=1000/fps, repeat=True
        )
        
        # Save as MP4
        output_path = Path(output_path)
        if output_path.suffix != '.mp4':
            output_path = output_path.with_suffix('.mp4')
        
        print(f"  Saving to: {output_path}")
        
        # FFmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, codec=codec, bitrate=1800)
        
        anim.save(str(output_path), writer=writer, dpi=100)
        plt.close()
        
        print(f"✓ MP4 video saved: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        return str(output_path)
    
    def save_frame_sequence(self, frames: List[Dict], output_dir: str,
                           show_metrics: bool = True, show_growth: bool = True) -> str:
        """
        Save individual PNG frames.
        
        Args:
            frames: List of frame dictionaries
            output_dir: Output directory for frames
            show_metrics: Show metrics overlay
            show_growth: Highlight new growth
        
        Returns:
            Path to output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving frame sequence...")
        print(f"  Output directory: {output_dir}")
        print(f"  Frames: {len(frames)}")
        
        previous_lcc = None
        
        for idx, frame in enumerate(frames, 1):
            fig, ax = plt.subplots(figsize=(12, 10))
            
            year = frame['year']
            
            # Create RGB image
            rgb = np.zeros((*frame['mask'].shape, 3), dtype=np.uint8)
            
            if show_growth and previous_lcc is not None:
                new_growth = (frame['lcc_mask'] == 1) & (previous_lcc == 0)
                rgb[new_growth] = [0, 255, 0]
                old_built = (previous_lcc == 1)
                rgb[old_built] = [200, 200, 200]
                rgb[frame['lcc_mask'] == 1] = [255, 100, 100]
            else:
                rgb[frame['mask'] == 1] = [220, 220, 220]
                rgb[frame['lcc_mask'] == 1] = [255, 50, 50]
            
            ax.imshow(rgb)
            ax.axis('off')
            
            # Title and metrics
            title = f'Year: {year}'
            if show_metrics:
                title += f'\nLCC Area: {frame["lcc_area_km2"]:.2f} km²  |  '
                title += f'Perimeter: {frame["perimeter_km"]:.1f} km  |  '
                title += f'Components: {frame["num_components"]}'
            
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            if show_metrics:
                metrics_text = (
                    f'LCC: {frame["lcc_size"]:,} pixels\n'
                    f'Perim: {frame["perimeter_pixels"]:,} pixels\n'
                    f'Coverage: {frame["lcc_percentage"]:.1f}%'
                )
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', bbox=props,
                       family='monospace')
            
            # Save frame
            frame_file = output_dir / f"frame_{idx:03d}_{year}.png"
            plt.savefig(frame_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved frame {idx}/{len(frames)}: {year}", end='\r')
            
            previous_lcc = frame['lcc_mask'].copy()
        
        print(f"\n✓ Frame sequence saved: {output_dir}")
        print(f"  Total frames: {len(frames)}")
        
        return str(output_dir)


def geocode_city(city_name: str) -> Tuple[float, float]:
    """Geocode city name"""
    print(f"Geocoding: {city_name}")
    geolocator = Nominatim(user_agent="wsf_animator")
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
        description='Create LCC evolution animation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create GIF animation
  python create_lcc_animation.py --lat 31.8122 --lon 119.9692 --radius 25 --output changzhou.gif
  
  # Create MP4 video (higher quality)
  python create_lcc_animation.py --city "Beijing, China" --radius 50 --output beijing.mp4 --fps 5
  
  # Save individual frames
  python create_lcc_animation.py --lat 31.8122 --lon 119.9692 --radius 25 --frames-only --output frames/
  
  # No growth highlighting
  python create_lcc_animation.py --city "Shanghai, China" --radius 40 --output shanghai.gif --no-growth
        """
    )
    
    parser.add_argument('--city', type=str, help='City name')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--radius', type=float, default=25, help='Download radius (km)')
    parser.add_argument('--size', type=float, help='Analysis size (km, default: 2×radius)')
    parser.add_argument('--output', type=str, required=True, help='Output file (.gif, .mp4) or directory (frames)')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second (default: 2)')
    parser.add_argument('--cache-dir', type=str, default='./wsf_cache', help='Tile cache directory')
    parser.add_argument('--no-metrics', action='store_true', help='Hide metrics overlay')
    parser.add_argument('--no-growth', action='store_true', help='Disable growth highlighting')
    parser.add_argument('--frames-only', action='store_true', help='Save only individual frames')
    parser.add_argument('--codec', type=str, default='libx264', help='Video codec for MP4 (default: libx264)')
    
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
    
    # Analysis size
    size_km = args.size if args.size else args.radius * 2
    
    print("="*70)
    print("LCC EVOLUTION ANIMATION CREATOR")
    print("="*70)
    print(f"\nLocation: {location_name}")
    print(f"Center: ({lat:.4f}, {lon:.4f})")
    print(f"Download radius: {args.radius} km")
    print(f"Analysis size: {size_km} km × {size_km} km")
    
    # Download tiles
    print("\n" + "="*70)
    print("DOWNLOADING TILES")
    print("="*70)
    
    manager = WSFTileManager(cache_dir=args.cache_dir)
    download_result = manager.download_region(lat, lon, args.radius)
    
    # Load and process data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    analyzer = BuiltAreaAnalyzer()
    data, metadata = analyzer.load_tiles_from_download_result(download_result)
    
    # Extract region
    data_subset, _ = analyzer.extract_built_area_bbox(
        data, metadata['transform'], lat, lon, size_km
    )
    
    # Create animator
    print("\n" + "="*70)
    print("CREATING ANIMATION")
    print("="*70)
    
    animator = LCCAnimator()
    frames = animator.create_animation_frames(data_subset, analyzer)
    
    # Generate output
    if args.frames_only:
        # Save individual frames
        animator.save_frame_sequence(
            frames, args.output,
            show_metrics=not args.no_metrics,
            show_growth=not args.no_growth
        )
    else:
        output_path = Path(args.output)
        
        if output_path.suffix == '.mp4':
            # Create MP4 video
            animator.create_video_animation(
                frames, args.output,
                fps=args.fps,
                show_metrics=not args.no_metrics,
                show_growth=not args.no_growth,
                codec=args.codec
            )
        else:
            # Create GIF (default)
            animator.create_gif_animation(
                frames, args.output,
                fps=args.fps,
                show_metrics=not args.no_metrics,
                show_growth=not args.no_growth
            )
    
    print("\n" + "="*70)
    print("✓ ANIMATION COMPLETE!")
    print("="*70)
    print(f"\nOutput: {args.output}")
    print(f"Frames: {len(frames)} (1985-2015)")
    print(f"\n💡 Tip: Open the file to view the animation!")


if __name__ == "__main__":
    main()