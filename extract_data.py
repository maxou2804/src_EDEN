from wsf_evolution_lcc import WSFTileManager, BuiltAreaAnalyzer, geocode_city

downloader = WSFTileManager(cache_dir="./wsf_cache")

lat,lon=geocode_city("changzhou")

tiles = downloader.calculate_required_tiles(lat, lon, radius_km=30)

print(tiles)

results= downloader.download_region(lat,lon, radius_km=30)


downloader.visualize_coverage(results=results)


analyzer = BuiltAreaAnalyzer()

analyzer.extract_built_area_bbox(raster_path=results, center_lat=lat,center_lon= lon, size_km=30)