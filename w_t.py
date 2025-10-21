from alpha_calculation_functions import extract_values_from_files, read_directory
import matplotlib.pyplot as plt
import numpy as np

## PLOT OF W AS A FUNCTION OF T  ###
directory="/Users/mika/Documents/PDM/outputs/study_binning/study_binning_grid_500/N_bin=200_V2"

N_sector, wt,radius,urban_fraction=read_directory(directory)

wt_ordered=wt[N_sector[:,1].argsort()]
N_sector_ordered=N_sector[N_sector[:,1].argsort()]

print(N_sector_ordered[:,1])
time=np.linspace(1,len(wt[1,:]),len(wt[1,:]))
k=range(0,len(wt[:,1]),1)

plt.figure()
for i in k:
    plt.loglog(time,wt_ordered[i],label=f"Data with N={N_sector_ordered[i,1]}")

plt.loglog(time,time**(1/3)*10**(-0.5))
plt.xlabel("time")
plt.ylabel('w')
plt.legend()
plt.show()



# l=2*np.pi*radius[i,:]/N_sector_ordered[i,1]















 