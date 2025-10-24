from alpha_calculation_functions import progressive_loglog_fit
import pandas as pd
import numpy as np
from numba import jit, prange
from typing import List, Tuple
import matplotlib.pyplot as plt
from perimeter_function import analyze_sectors_optimized
from collapse_functions import tolerant_mean


wt_collection=[]
l_collection=[]
beg_points_to_skip=0
end_points_to_skip=1


file_name="perimeter_mexico_city.csv"
#calculate N_sectors
min_N = 5
max_N = 2000
num_N=30
N_list = np.logspace(np.log10(min_N), np.log10(max_N), num_N, dtype=int)


year_list=[1985,2000,2005,2010]


output=analyze_sectors_optimized(file_name,N_sector=N_list,years=year_list)



for i in range(0,2*len(year_list),2):
    
    l=2*np.pi/N_list*output[i+1,:]
    w=output[i,:]
    indices = np.argsort(l)
    l=l[indices]
    w=w[indices]
    l_collection.append(l)
    wt_collection.append(w)
print(wt_collection)

w_avg,w_std=tolerant_mean(wt_collection)
l_avg,l_std=tolerant_mean(l_collection)

print(w_avg)

result = progressive_loglog_fit(l_avg, w_avg,beg_points_to_skip,end_points_to_skip, std_threshold=0.01)


print(f"Slope = {result['slope']:.3f} ± {result['slope_std']:.3f}")
print(f"Used {result['used_points']} points")

plt.loglog(l_avg,w_avg,'o')
plt.loglog(l_avg[beg_points_to_skip:-end_points_to_skip],l_avg[beg_points_to_skip:-end_points_to_skip]**result["slope"]*10**result["intercept"], '-', label=f"slope : {result['slope']:.3f}")




plt.xlabel("l")
plt.ylabel("w")
plt.legend()
plt.show()






