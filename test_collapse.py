import numpy as np
import matplotlib.pyplot as plt
from collapse_functions import minimize_collapse_normalized,collapse_error_normalized,grid_error_map, plot_before_after
from alpha_calculation_functions import extract_values_from_files

# --- True exponents (KPZ universality class) ---
beta_true = 1/3
inv_z_true = 2/3

# --- Population values ---
P_list = [50, 100, 200, 400, 800]

# --- Scaling function F(x): saturates for large x ---
def F(x):
    return np.tanh(x)**0.5

# --- Generate synthetic curves ---
x_list, y_list = [], []
for P in P_list:
    l = np.logspace(0, 3, 500)             # "length scale" axis
    w = (P**beta_true) * F(l / (P**inv_z_true))
    w *= 1 + 0.01*np.random.randn(len(l))  # add 5% noise
    x_list.append(l)
    y_list.append(w)

# --- Try collapse ---
res = minimize_collapse_normalized(x_list, y_list, P_list,
                                   beta_guess=0.3, inv_z_guess=0.6)
print("Recovered:", res)

plot_before_after(x_list, y_list, P_list, res['beta'], res['inv_z'])

# --- Grid landscape for inspection ---
beta_vals = np.linspace(0.1, 0.6, 30)
inv_z_vals = np.linspace(0.3, 1.0, 30)
E = grid_error_map(x_list, y_list, P_list, beta_vals, inv_z_vals)

plt.figure(figsize=(6,5))
plt.contourf(inv_z_vals, beta_vals, E, 30, cmap='viridis')
plt.colorbar(label="collapse error")
plt.xlabel("1/z"); plt.ylabel("β")
plt.scatter(res['inv_z'], res['beta'], c="r", s=50, label="fit")
plt.legend()
plt.show()
