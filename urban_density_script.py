

from wsf_evolution_lcc import WSFTileManager, BuiltAreaAnalyzer, geocode_city

downloader = WSFTileManager(cache_dir="./wsf_cache")


name_city="Bangkok"
output_path="/Users/mika/Documents/PDM/src_EDEN_git/src_EDEN-1/masks/Mexico City"

lat,lon=geocode_city(name_city)
# calculates the required tiles based on the position and the radius
tiles = downloader.calculate_required_tiles(lat, lon, radius_km=100)

# downloads the corresponding tiles
results= downloader.download_region(lat,lon, radius_km=100)

# vizualisze the tiles if necessary
#downloader.visualize_coverage(results=results)



# create the built analzyer
analyzer = BuiltAreaAnalyzer()

data, metadata = analyzer.load_tiles_from_download_result(results)
from urban_density_calculator import calculate_urban_density_timeseries




density_df = calculate_urban_density_timeseries(
    wsf_data=data,
    analyzer=analyzer,
    radius_factor=4,
    years=list(range(1985, 1986))
)

# Save results
density_df.to_csv(f"urban_density_timeseries_{name_city}.csv", index=False)

# # Step 4: Visualize the calculation
# from urban_density_calculator import visualize_density_calculation

# visualize_density_calculation(wsf_data=data,
#                                   year=1985,
#                                   analyzer=analyzer,  
#                                   radius_factor = 4.0,
#                                   output_path=f"urban_visualization_{name_city}.png",
#                                   zoom_factor= 1.2,
#                                   n_clusters = 10,
#                                   show_cluster_labels= True)





# from urban_density_calculator import track_urban_clusters_growth

# growth_results = track_urban_clusters_growth(
#     wsf_data=data,
#     analyzer=analyzer,
#     n_clusters=5,  # Track LCC + 5 next largest clusters
#     start_year=1985,
#     end_year=2015
# )

# # Access summary data
# summary_df = growth_results['summary']
# print(summary_df)

# # Access time series data
# timeseries_df = growth_results['timeseries']
# print(timeseries_df)

# # Step 6: Visualize cluster growth
# from urban_density_calculator import visualize_clusters_growth

# visualize_clusters_growth(
#     growth_results=growth_results,
#     output_path=f"clusters_growth_analysis_{name_city}.png",
#     show_top_n=6  # Show all tracked clusters
# )

# # Step 7: Export comprehensive report
# from urban_density_calculator import export_clusters_growth_report

# files = export_clusters_growth_report(
#     growth_results=growth_results,
#     output_dir="./results",
#     city_name=f"{name_city}"
# )

# print("Files created:", files)

# # Step 8: Visualize clusters for a single year
# from urban_density_calculator import visualize_clusters_map

# visualize_clusters_map(
#     wsf_data=data,
#     year=1985,
#     analyzer=analyzer,
#     n_clusters=10,           # Show top 10 clusters
#     center_on_lcc=True,      # Center view on LCC
#     zoom_factor=1.5,         # 50% buffer around LCC
#     show_labels=True,        # Show cluster labels
#     output_path=f"clusters_map_1985_{name_city}.png"
# )

# Step 9: Create animated GIF showing evolution (1985-2015)
#from urban_density_calculator import create_clusters_animation

# create_clusters_animation(
#     wsf_data=data,
#     analyzer=analyzer,
#     output_path=f"clusters_evolution_{name_city}.gif",
#     start_year=1985,
#     end_year=2015,
#     n_clusters=10,           # Track top 10 clusters
#     center_on_lcc=True,      # Keep view centered on LCC
#     zoom_factor=1.5,         # View extent
#     show_labels=True,        # Show cluster labels on each frame
#     fps=3,                   # 2 frames per second (slow)
#     keep_frames=False        # Don't keep individual PNG frames
# )