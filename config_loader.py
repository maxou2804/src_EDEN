import os
import yaml

# Default configuration
DEFAULT_CONFIG = {
    "grid_size": 100,
    "threshold": 0.6,
    "radius": 10,
    "urbanization_prob": 0.25,
    "timesteps": 20,
    "distance_decay": False,
    "initial_core": "circle",
    "sampling":0.5,
    "k":5
}

def load_config(path="C:\\Users\\trique\\Downloads\\MASTER_THESIS\\src_EDEN\\config.yaml"):
    """
    Load YAML configuration with fallback to default values.
    - If file does not exist, returns DEFAULT_CONFIG.
    - If some keys are missing, fills in defaults.
    """
    # Resolve absolute path relative to this file
    abs_path = os.path.join(os.path.dirname(__file__), path)

    if not os.path.isfile(abs_path):
        print(f"Warning: Config file '{abs_path}' not found. Using default configuration.")
        return DEFAULT_CONFIG.copy()

    with open(abs_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return DEFAULT_CONFIG.copy()

    # Fill missing keys with defaults
    final_config = DEFAULT_CONFIG.copy()
    final_config.update(config)

    return final_config
