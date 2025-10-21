# Generate 10 logarithmically spaced numbers between 10^20 and 10^100
import numpy as np

result = np.logspace(np.log10(20),np.log10(150), 20)

int_array=result.astype(np.int32)

print("prout")




