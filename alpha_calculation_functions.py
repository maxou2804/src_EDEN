import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression


import os
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np


def read_csv_at_time(filename, given_time, col_time="time", col_w="w", col_radius="mean_radius",
                     col_urban="urban_fraction", col_N="N"):
    """
    Reads a CSV file and extracts all rows for a given time,
    collecting values of w, mean radius, urban_fraction, and N.

    Parameters
    ----------
    filename : str
        The path to the CSV file.
    given_time : int
        The time value to filter rows.
    col_time : str
        Column name for time.
    col_w : str
        Column name for w.
    col_radius : str
        Column name for mean radius.
    col_urban : str
        Column name for urban_fraction.
    col_N : str
        Column name for N.

    Returns
    -------
    tuple of np.ndarray and float
        (N_array, w_array, radius_array, urban_fraction_scalar)
        Returns empty arrays and 0.0 if no data for the time.
    """
    df = pd.read_csv(filename)
    index=int(given_time*len(df[col_w]))-1
    time_index=df.at[index,"time"]
    #print(f"Time index is : {time_index}")

    filtered = df[df['time']==time_index]
    if not filtered.empty:
        filtered = filtered.sort_values(col_N)
        N_array = filtered[col_N].values
        w_array = filtered[col_w].values
        radius_array = filtered[col_radius].values
        urban_fraction_scalar = filtered[col_urban].values[0]  # Same for all rows
    else:
        print(f"No rows for time {given_time} in {filename}.")
        N_array = np.array([])
        w_array = np.array([])
        radius_array = np.array([])
        urban_fraction_scalar = 0.0

    return N_array, w_array, radius_array, urban_fraction_scalar



def read_directory(directory, col_wt="w_t", col_radius="mean_radius",col_urban="urban_fraction",col_N="n_sectors"):
    """
    Reads all CSV files in a directory, extracts a chosen row from each file,
    and collects values of w_t and mean radius into arrays.

    Parameters
    ----------
    directory : str
        Path to directory containing CSV files.
    row_number : int
        The row index (0-based) to extract from each file.
    col_wt : str
        Column name for w_t.
    col_radius : str
        Column name for mean radius.

    Returns
    -------
    tuple of np.ndarray
        (w_t_array, radius_array) for all files
    """
    wt_values = []
    radius_values = []
    urban_fraction=[]
    N_values=[]

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            
            urban_fraction.append(df[col_urban])
            wt_values.append(df[col_wt])
            radius_values.append(df[col_radius])
            N_values.append(df[col_N])
            
    return np.array(wt_values),np.array(urban_fraction), np.array(radius_values), np.array(N_values)




def progressive_loglog_fit(x, y, a, b, std_threshold=0.05, min_points=5):

    # Slice to cut out first a and last b points
    x = np.asarray(x)[a: -b if b > 0 else None]
    y = np.asarray(y)[a: -b if b > 0 else None]

    # log-transform
    logx = np.log10(x)
    logy = np.log10(y)

    for n in range(min_points, len(x)+1):
        X = logx[:n].reshape(-1, 1)
        Y = logy[:n]
        
        model = LinearRegression().fit(X, Y)
        slope = model.coef_[0]
        intercept = model.intercept_

        residuals = Y - model.predict(X)
        slope_std = np.std(residuals) / np.sqrt(len(residuals))

        if slope_std > std_threshold:
            # stop before bad region
            n_used = n-1
            break
    else:
        n_used = len(x)

    # final fit on valid region
    X = logx[:n_used].reshape(-1, 1)
    Y = logy[:n_used]
    model = LinearRegression().fit(X, Y)

    slope = model.coef_[0]
    intercept = model.intercept_
    residuals = Y - model.predict(X)
    slope_std = np.std(residuals) / np.sqrt(len(residuals))

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_std": slope_std,
        "used_points": n_used,
      
    }