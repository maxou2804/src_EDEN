import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label, generate_binary_structure, gaussian_filter, binary_dilation

# ---------------- Parameters ----------------
size = 100
threshold = 0.99
radius = size // 20
urbanization_prob = 0.005
timesteps = 200
distance_decay = True

# ---------------- Initial setup ----------------
grid = np.random.rand(size, size)

# Initial city core
city_seed = np.zeros((size, size))
cx, cy = size // 2, size // 2
for x in range(size):
    for y in range(size):
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if dist < radius + np.random.randint(-3, 3):
            city_seed[x, y] = 1

from scipy.ndimage import gaussian_filter
city_seed = gaussian_filter(city_seed, sigma=2)
city_core = (city_seed > 0.3).astype(int)

# Initial urbanization
urban = (grid > threshold).astype(int)
urban = np.maximum(urban, city_core)

# Classification function
def classify(urban, city_core):
    structure = generate_binary_structure(2, 2)
    labeled, num_clusters = label(urban, structure=structure)
    classified = np.zeros_like(urban)
    if num_clusters > 0:
        core_labels = np.unique(labeled[city_core == 1])
        core_labels = core_labels[core_labels > 0]
        for lbl in range(1, num_clusters + 1):
            if lbl in core_labels:
                classified[labeled == lbl] = 2
            else:
                classified[labeled == lbl] = 1
    return classified

# Simulation generator (yields each timestep)
def simulate(urban, city_core, timesteps, prob, decay=False):
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    dist_grid = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.max(dist_grid)

    for t in range(timesteps):
        classified = classify(urban, city_core)
        yield classified  # <-- give current state to animation

        # Boundary detection
        structure = generate_binary_structure(2, 2)
        boundary_mask = binary_dilation(urban, structure=structure) & (urban == 0)

        # Growth probability
        if decay:
            growth_prob = prob * (1 - dist_grid / max_dist)
            growth_prob = np.clip(growth_prob, 0.01, 1.0)
        else:
            growth_prob = prob

        grow_mask = (np.random.rand(size, size) < growth_prob) & boundary_mask
        urban[grow_mask] = 1

        # Update city_core persistently
        classified = classify(urban, city_core)
        city_core = (classified == 2).astype(int)

# ---------------- Animation ----------------
cmap = ListedColormap(["white", "royalblue", "crimson"])

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(np.zeros((size, size)), cmap=cmap, vmin=0, vmax=2)
ax.set_title("Urban sprawl simulation")
ax.axis("off")

# Update function
def update(frame):
    im.set_data(frame)
    return [im]

# Build animation
frames = simulate(urban, city_core, timesteps, urbanization_prob, decay=distance_decay)
ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=500, repeat=False)

plt.show()

# To save as GIF (uncomment if needed):
# ani.save("urban_sprawl.gif", writer="pillow", fps=2)
