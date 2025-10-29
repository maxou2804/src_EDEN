import numpy as np




high_beta=[55.01,51.84,70.97,58.63,65.37]
low_beta=[44.63,54.67,35.50,22.31,55.63]

mean_high=np.mean(high_beta)
mean_low=np.mean(low_beta)


std_high=np.std(high_beta)
std_low=np.std(low_beta)    

print("High beta mean:", mean_high)
print("Low beta mean:", mean_low)
print("High beta std:", std_high)
print("Low beta std:", std_low)               