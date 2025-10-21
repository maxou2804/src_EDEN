import itertools
from cycler import cycler
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import numpy as np
from collapse_functions import minimize_collapse_normalized,collapse_error_normalized,grid_error_map, plot_before_after, tolerant_mean, add_array_with_padding
from alpha_calculation_functions import read_directory, read_csv_at_time





directory="C:\\Users\\trique\\Downloads\\MASTER_THESIS\\outputs\\grid_runs_beta_z\\simul_L_2000"  

wt_avg_collection=[]
l_avg_collection=[]
urban_avg_collection=[]

# time at which we do the alpha calculation 
# time=-1

wt_collection=[]
l_collection=[]
urban_collection=[]


time_serie= np.linspace(0.20,0.9,5)


for time in time_serie:
    print("wesh")

    for filename in os.listdir(directory):
        filepath=os.path.join(directory,filename)
        N_sector, wt,radius,urban_fraction=read_csv_at_time(filepath,time)


        l=2*np.pi*radius/N_sector
        indices = np.argsort(l)

        l=l[indices]
        wt=wt[indices]

        wt_collection.append(wt)
        l_collection.append(l)
        urban_collection.append(urban_fraction)

    l_avg,l_std=tolerant_mean(l_collection)
    wt_avg,w_std=tolerant_mean(wt_collection)
    urban_avg=np.mean(urban_collection)



    wt_collection=[]
    l_collection=[]
    urban_collection=[]
  
    

    wt_avg_collection=add_array_with_padding(wt_avg_collection, wt_avg)
    l_avg_collection= add_array_with_padding(l_avg_collection,  l_avg)
   
    urban_avg_collection.append(np.mean(urban_avg))



    






    #plt.loglog(x_sorted*P_sorted**(-2/3), y_sorted*P_sorted**(-1/3), 'o', label=f"data t={time}")
    #plt.loglog(x_sorted*P_sorted**(1/3), y_sorted*P_sorted**(-2/3), 'o', label=f"data t={time}")

    #plt.loglog(x_sorted*P_sorted**(1/3), y_sorted*P_sorted**(-2/3), 'o', label=f"data t={time}")
    plt.loglog(l_avg/urban_fraction**(2/3),wt_avg/urban_fraction**(1/3), 'o', label=f"data t={time}")
    #plt.loglog(l_avg,wt_avg, 'o', label=f"data t={time}")    
    # plt.loglog(l/urban_fraction**(2/3),wt/urban_fraction**(1/3),'o',label=f"data collapsed t={time}")
    



plt.xlabel('l*P^(-1/z)')
plt.ylabel('w*P^(-beta)')
plt.legend()
plt.grid()
plt.show()






 



beta_vals = np.linspace(0.0, 1, 50)   # adjust
inv_z_vals = np.linspace(0, 1, 50)
E = grid_error_map(l_avg_collection, wt_avg_collection, urban_avg_collection, beta_vals, inv_z_vals, n_interp=200)

# find min
min_idx = np.unravel_index(np.nanargmin(E), E.shape)
print("Grid best:", beta_vals[min_idx[0]], inv_z_vals[min_idx[1]], "err =", E[min_idx])

# plot heatmap
plt.figure(figsize=(6,5))
plt.contourf(inv_z_vals, beta_vals, E, levels=40, cmap='viridis')
plt.xlabel("1/z"); plt.ylabel("beta")
plt.colorbar(label="normalized collapse error")
plt.scatter([inv_z_vals[min_idx[1]]], [beta_vals[min_idx[0]]], color='red')
plt.gca().invert_yaxis()  # optional
plt.show()

res = minimize_collapse_normalized(l_avg_collection, wt_avg_collection, urban_avg_collection,
                                   beta_guess=0.3, inv_z_guess=0.6,
                                   bounds=((0.0,2),(0.01,2)),
                                   n_interp=200)
print(res)

plot_before_after(l_avg_collection, wt_avg_collection, urban_avg_collection, res['beta'], res['inv_z'])

