import os
import numpy as np
import pandas as pd

def save_arrays(arrays_dict, out_dir, base_name="results"):
    """
    Save arrays into a single CSV file.

    Parameters
    ----------
    arrays_dict : dict
        Dictionary where keys are names (str) and values are 1D numpy arrays.
        Arrays must all have the same length.
    out_dir : str
        Path to the output directory.
    base_name : str
        Base name of the output file.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Convert each array to a pandas Series to avoid ordering issues
    series_dict = {}
    for name, arr in arrays_dict.items():
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(f"Array '{name}' must be 1D, got shape={arr.shape}")
        series_dict[name] = pd.Series(arr)

    # Combine into DataFrame
    df = pd.DataFrame(series_dict)

    # Save to CSV
    out_path = os.path.join(out_dir, f"{base_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved arrays to {out_path}")
