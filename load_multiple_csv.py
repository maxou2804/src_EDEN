import numpy as np
import pandas as pd
import os

def load_multiple_csv(base_path, base_name, ext="csv", indices=range(1, 6)):
    """
    Load multiple CSV files into one pandas DataFrame.

    Parameters
    ----------
    base_path : str
        Directory where files are located.
    base_name : str
        Common prefix of the file names.
    ext : str
        File extension (default: 'csv').
    indices : iterable
        List/range of indices appended to base_name.

    Returns
    -------
    #/Users/mika/Documents/PDM/outputs/critical_exponent
    pd.DataFrame
        Concatenated dataframe with an extra column 'run_id'.
    """
    dfs = []
    for i in indices:
        file_path = os.path.join(base_path, f"{base_name}{i}.{ext}")
        df = pd.read_csv(file_path)
        df["run_id"] = i   # add a column to keep track of which file
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
