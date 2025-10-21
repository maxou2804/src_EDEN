import numpy as np


p_0_2=[0.7037,0.8113,0.7864,0.9445,0.9228,0.8717]

p_0_1=[0.8977,0.7831,0.8112,0.8649,0.9226,0.7244]

p_00_5=[0.9586,1.003,0.9633,0.9414,0.979,0.852]




print("p_urb=0.05 mean an variance")
print(np.mean(p_00_5))
print(np.var(p_00_5))

print("p_urb=0.1 mean an variance")
print(np.mean(p_0_1))
print(np.var(p_0_1))

print("p_urb=0.2 mean an variance")
print(np.mean(p_0_2))
print(np.var(p_0_2))





initial_10=[1.111,1.132,1.036,0.9958,0.9644,1.037]
print("initial size 10% mean and variance")
print(np.mean(initial_10))
print(np.var(initial_10))