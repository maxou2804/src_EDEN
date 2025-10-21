import numpy as np
from scipy.sparse import issparse

def calculate_w_2(front_collection, N):
    w_collection = np.zeros((len(front_collection),))
    mean_collection = np.zeros((len(front_collection),))

    for i, front in enumerate(front_collection):
        r_collection = radial_profile(front, N)

        std = np.array([r["std_sector"] for r in r_collection])
        mean = np.array([r["mean_radius"] for r in r_collection])

        w_collection[i] = 1 / N * np.sum(std)
        mean_collection[i] = 1 / N * np.sum(mean)

    return w_collection, mean_collection


def radial_profile(front_mask, n_sectors):
    """
    Compute mean radius and local fluctuations of an urban front per sector.

    Parameters
    ----------
    front_mask : 2D array (0/1) or sparse matrix
        Binary mask where 1 = front pixel, 0 = non-front.
    n_sectors : int
        Number of angular sectors (slices of 2π/N).

    Returns
    -------
    stats : list of dict
        Each dict contains {"sector", "mean_radius", "std_sector"}.
    """
    # Extract coordinates of front pixels
    if issparse(front_mask):
        ys, xs = front_mask.nonzero()
    else:
        ys, xs = np.where(front_mask == 1)

    if len(xs) == 0:
        raise ValueError("Front mask is empty — no pixels with value 1.")

    # Center of mass (manual, since sparse does not support ndimage.center_of_mass)
    cy = ys.mean()
    cx = xs.mean()

    # Distances from center
    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # Angles of each pixel (in radians)
    angles = (np.arctan2(ys - cy, xs - cx) + 2 * np.pi) % (2 * np.pi)
    sector_ids = (angles / (2 * np.pi / n_sectors)).astype(int)

    # Collect stats per sector
    stats = []
    for i in range(n_sectors):
        sector_distances = distances[sector_ids == i]
        if len(sector_distances) > 0:
            mean_r = sector_distances.mean()
            std_r = np.mean((sector_distances - mean_r) ** 2) ** 0.5  # std
        else:
            mean_r = 0.0
            std_r = 0.0
        stats.append({
            "sector": i,
            "mean_radius": mean_r,
            "std_sector": std_r
        })

    return stats





def sector_masks(shape, center, n_sectors):
    """
    Create boolean masks dividing the grid into N circular sectors
    that extend to the grid edges.

    Parameters
    ----------
    shape : tuple
        Shape of the 2D grid (height, width).
    center : tuple
        (cy, cx) coordinates of the circle center.
    n_sectors : int
        Number of angular sectors.

    Returns
    -------
    masks : list of 2D boolean arrays
        Each mask corresponds to one sector.
    """
    Y, X = np.ogrid[:shape[0], :shape[1]]
    cy, cx = center

    # Coordinates relative to center
    dx = X - cx
    dy = Y - cy

    # Compute angles in degrees (0° = positive x-axis)
    theta = np.degrees(np.arctan2(dy, dx)) % 360

    # Build masks for each sector (covering the full grid)
    masks = []
    sector_angle = 360 / n_sectors
    for i in range(n_sectors):
        theta_min = i * sector_angle
        theta_max = (i + 1) * sector_angle
        mask = (theta >= theta_min) & (theta < theta_max)
        masks.append(mask)

    return masks
