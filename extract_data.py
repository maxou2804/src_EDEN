from wsf_evolution_lcc import WSFTileManager, BuiltAreaAnalyzer, geocode_city, export_lcc_coordinates_all_years
import pandas as pd
from perimeter_function import extract_perimeter_from_bbox_optimized
# create the directory where to store tiles
downloader = WSFTileManager(cache_dir="./wsf_cache")

# find the lat and lon of the city trough its name


name_city="Beijing"
output_path="/Users/mika/Documents/PDM/src_EDEN_git/src_EDEN-1/masks/Mexico_City"

lat,lon=geocode_city(name_city)
# calculates the required tiles based on the position and the radius
tiles = downloader.calculate_required_tiles(lat, lon, radius_km=50)

# downloads the corresponding tiles
results= downloader.download_region(lat,lon, radius_km=50)

# vizualisze the tiles if necessary
#downloader.visualize_coverage(results=results)



# create the built analzyer
analyzer = BuiltAreaAnalyzer()

data, metadata = analyzer.load_tiles_from_download_result(results)

data_subset, meta_subset = analyzer.extract_built_area_bbox(
    data=data,
    transform=metadata['transform'],
    center_lat=31.8122,
    center_lon=119.9692,
    size_km=50



)

data=extract_perimeter_from_bbox_optimized(data_subset,meta_subset['transform'],10000 ,use_numba=True)  
data.to_csv('perimeter_beijing.csv', index=False)