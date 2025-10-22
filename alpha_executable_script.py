import numpy as np
import pandas as pd
import os
from load_multiple_csv import load_multiple_csv
from fitting import find_best_collapse, plot_collapse
from alpha_calculation_functions import read_csv_at_time, progressive_loglog_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from collapse_functions import tolerant_mean

# Generate 10 logarithmically spaced numbers between 10^20 and 10^100


wt_collection=[]
l_collection=[]

# directory="/Users/mika/Documents/PDM/outputs/13_10_25"
directory="C:\\Users\\trique\\Downloads\\MASTER_THESIS\\outputs\\grid_runs_test"
#time at which we do the alpha calculation (give 0 to 1)
time_extract=0.9

#give points to skip for the fit
beg_points_to_skip=10
end_points_to_skip=5

#extract values from cvs

for filename in os.listdir(directory):

    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        #print(filename)
        N_sector, wt,radius,urban_fraction=read_csv_at_time(filepath,time_extract)

        l=2*np.pi*radius/N_sector
        indices = np.argsort(l)
        l=l[indices]
        wt=wt[indices]
        l_collection.append(l)
        wt_collection.append(wt)
     
    
#wt_std=np.std(wt_collection,axis=0)
#print(wt_std)
wt_avg,wt_std=tolerant_mean(wt_collection)
l_avg,l_std=tolerant_mean(l_collection)





# print(wt)
# print(radius)
# print(urban_fraction)
# print(N_sector)




result = progressive_loglog_fit(l_avg, wt_avg,beg_points_to_skip,end_points_to_skip, std_threshold=0.01)


print(f"Slope = {result['slope']:.3f} ± {result['slope_std']:.3f}")
print(f"Used {result['used_points']} points")


n=result["used_points"]

# Plot
plt.loglog(l_avg, wt_avg, 'o', label=f"data t={time_extract}")
plt.loglog(l_avg, wt_avg+wt_std)
plt.loglog(l_avg, wt_avg-wt_std)
plt.loglog(l_avg[beg_points_to_skip:-end_points_to_skip],l_avg[beg_points_to_skip:-end_points_to_skip]**result["slope"]*10**result["intercept"], '-', label=f"slope : {result['slope']:.3f}")
plt.xlabel("l")
plt.ylabel("w")
plt.legend()
#plt.savefig("/Users/mika/Documents/PDM/outputs/alpha_fit")
plt.show()


















# # df=load_multiple_csv(filepath,"results_k=","csv",range(4))




# # res = find_best_collapse(df)





# # print("BEST:", res["best_beta"], res["best_z"], "error:", res["best_error"])

# # # plot collapse
# # plot_collapse(df, res, beta=res["best_beta"], z=res["best_z"])







