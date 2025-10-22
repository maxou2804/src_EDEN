import numpy as np
from core import create_city_core
from simulation import simulate
from viz import plot_metrics, animate
from config_loader import load_config
import matplotlib.pyplot as plt
from calculate_w import calculate_w
from calculate_w_2 import calculate_w_2
from save_arrays import save_arrays
import os

# Load parameters







# === CONFIGURABLE PARAMETERS ===

grid_sizes = [2000]
num_simulations_per_size = 1  # 🟡 CHANGE THIS to control how many runs per grid size

threshold = 2
urbanization_prob = 1
distance_decay = False
core_shape = "circle"
sampling = 1
k = 30

def compute_timesteps(L):
    # 🟢 Replace with your actual expression
    return int(((L/2)**2-(L/10)**2)*np.pi*0.9)


# Output base directory
base_output_dir = "C:\\Users\\trique\\Downloads\\MASTER_THESIS\\outputs\\grid_runs_test"
os.makedirs(base_output_dir, exist_ok=True)

# === RUN SIMULATIONS ===

for size in grid_sizes:
    radius = size // 10
    timesteps = compute_timesteps(size)
    metric_timesteps=np.logspace(4, 6.3, 100)

    print(f"\n📦 Grid size: {size}x{size} | Radius: {radius} | Timesteps: {timesteps}")

    for run_id in range(1, num_simulations_per_size + 1):
        print(f" 🔁 Simulation run {run_id}/{num_simulations_per_size} for size {size}")

        # Random grid and city core
        grid = np.random.rand(size, size)
        city_core = create_city_core(size, radius, shape=core_shape)

        # Initialize urban array
        urban_array = (grid > threshold).astype(int)
        urban_array = np.maximum(urban_array, city_core)

        # Save result to CSV
        output_file = os.path.join(
            base_output_dir, f"simul_L_{size}_run_{run_id}.csv"
        )
     

        # Run simulation
        result = simulate(
            grid_size=size,
            urban=urban_array,
            city_core=city_core,
            timesteps=timesteps,
            k=k,
            prob=1,
            sampling=1,
            adaptive_binning=True,
            use_binning=False,
            output_file=output_file,
            batch_size_param=0,
            batch_size_mode='single',
            metric_interval=100000,
            visualize_interval=50000,
            max_bins=20000)
          
        print(f"    📝 Results saved to {output_file}" )

      

print("\n✅ All simulations completed.")


