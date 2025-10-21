import numpy as np
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
from abc import ABC, abstractmethod


# -------------------------------
# Base Class for City Core
# -------------------------------
class CityCoreInitializer(ABC):
    def __init__(self, size, radius):
        self.size = size
        self.radius = radius
        self.cx, self.cy = size // 2, size // 2

    @abstractmethod
    def generate(self):
        """Return a 2D numpy array (0/1) for the city core mask"""
        pass

    def _smooth(self, mask):
        """Apply Gaussian smoothing and binarize"""
        mask = gaussian_filter(mask, sigma=2)
        return (mask > 0.3).astype(int)


# -------------------------------
# Different Core Shapes
# -------------------------------
class CircleCore(CityCoreInitializer):
    def generate(self):
        mask = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                dist = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
                if dist < self.radius:
                    mask[x, y] = 1
        return self._smooth(mask)


class SquareCore(CityCoreInitializer):
    def generate(self):
        mask = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if abs(x - self.cx) < self.radius and abs(y - self.cy) < self.radius:
                    mask[x, y] = 1
        return self._smooth(mask)


class EllipseCore(CityCoreInitializer):
    def generate(self):
        mask = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                dx, dy = x - self.cx, y - self.cy
                if (dx**2) / (self.radius**2) + (dy**2) / ((self.radius * 1.5)**2) < 1:
                    mask[x, y] = 1
        return self._smooth(mask)


class DiamondCore(CityCoreInitializer):
    def generate(self):
        mask = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                dx, dy = abs(x - self.cx), abs(y - self.cy)
                if dx + dy < self.radius:
                    mask[x, y] = 1
        return self._smooth(mask)


class SinglePointCore(CityCoreInitializer):
    def generate(self):
        """Generate a city core with a single point at the grid center."""
        mask = np.zeros((self.size, self.size))
        mask[self.cx, self.cy] = 1
        return self._smooth(mask)


# -------------------------------
# Factory function
# -------------------------------
def create_city_core(size, radius, shape="circle"):
    shape_map = {
        "circle": CircleCore,
        "square": SquareCore,
        "ellipse": EllipseCore,
        "diamond": DiamondCore,
        "single_point": SinglePointCore
    }
    if shape not in shape_map:
        raise ValueError(f"Unknown core shape '{shape}'. Available: {list(shape_map.keys())}")
    return shape_map[shape](size, radius).generate()


# -------------------------------
# Classification (same as before)
# -------------------------------
def classify(urban, city_core):
    """
    0 = non-urban
    1 = clusters not connected to city core
    2 = city core
    """
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