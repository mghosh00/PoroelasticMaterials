"""
In this file, we compare the analytic steady state to the numerical steady state
from one of our simulations. This steady state assumes that the Young's modulus
has reached a uniform steady state of E_min (so this is long-term analysis). We produce
comparative plots (and calculate a sum of squares error).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

from steady_state import SteadyState

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

# Whether we use an iterative solver or not
iterative = False

"""
Reading in our parameters and defining our paths
"""
trial = "long_steady_state"
sub_trial = "v_0_1"
dir_path = f"resources/{trial}/{sub_trial}"
data_path = f"{dir_path}/data"
plot_path = f"{dir_path}/plots"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)

# The parameters from the simulation
N_x = params["comp"]["N_x"]

phi_f0 = params["ics"]["phi_f"]

param_file.close()


def final_array(path: str):
    """Gets the final profile for the quantity from the given filepath.

    :param path: The path to the .csv file containing time series data for the quantity.
    :return: The np.array of the quantity at the last timestep.
    """
    f_array = pd.read_csv(path).to_numpy()
    f_final = f_array[0:, -1]
    return f_final[~np.isnan(f_final)]


# Find the final value of a(t) --- this comes from the u_s_array
u_s_final = final_array(f"{data_path}/_u_s_xi.csv")
a_final = u_s_final[0]

phi_f_final = final_array(f"{data_path}/_phi_xi.csv")
c_final = final_array(f"{data_path}/_c_xi.csv")
x = np.linspace(1 - (len(phi_f_final) - 1) / N_x, 1, len(phi_f_final))
xi = np.linspace(0, 1, N_x + 1)

# Use the custom SteadyState class
steady_state = SteadyState(params, xi)

# phi_l = get_phi_l(sigma_l, E_min, phi_f0, nu)
# factor = (t_phi * v_final * phi_f0 ** 3) / (t_v * E_min * (1 - phi_f0))
# We need a loop to converge on solutions for phi_f, a and B
phi_f_guess = np.array([phi_f0] * len(xi))

if iterative:
    max_its = 100
    phi_f_ss, a_ss, B_ss = steady_state.solve_iterative(phi_f_guess, max_its)
else:
    phi_f_ss, a_ss, B_ss = steady_state.solve_analytic()

c_ss = steady_state.calculate_c(phi_f_ss, a_ss)
u_s_ss = steady_state.calculate_u_s(phi_f_ss, a_ss)

alternative_B_ss = steady_state.alternative_B()

# Finally, we plot the prediction and the actual values of the final state for the
# Young's modulus. We'll also plot the linearised version of phi alongside the actual
# phi profile.

nrows, ncols = 3, 1
# nrows, ncols = 1, 1
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(8, nrows * 10/3))
ax_phi, ax_c, ax_us = axs
# ax_phi = axs

label_ss = "Iterated steady state" if iterative else "Analytical steady state"

# Plotting for phi
ax_phi.plot(xi, phi_f_final, "-k", label="Final profile from numerics")
ax_phi.plot(xi, phi_f_ss, "--r", label=label_ss)
ax_phi.legend()
ax_phi.set_xlabel("$\\xi$")
ax_phi.set_ylabel("$\\phi_f$")

# Plotting for c
ax_c.plot(xi, c_final, "-k", label="Final profile from numerics")
ax_c.plot(xi, c_ss, "--r", label=label_ss)
ax_c.legend()
ax_c.set_xlabel("$\\xi$")
ax_c.set_ylabel("$c$")

# Plotting for u_s
ax_us.plot(xi, u_s_final, "-k", label="Final profile from numerics")
ax_us.plot(xi, u_s_ss, "--r", label=label_ss)
ax_us.legend()
ax_us.set_xlabel("$\\xi$")
ax_us.set_ylabel("$u_s$")

# Saving the figure
it_text = "_it" if iterative else ""
fig.savefig(f"{plot_path}/_long_steady_state{it_text}.png", bbox_inches="tight")

print(f"True value of a_inf: {a_final}")
print(f"Predicted value of a_inf: {a_ss}")
print(f"Predicted value of B: {B_ss}")
print(f"Critical value of B: {alternative_B_ss}")

gamma_crit = steady_state.find_gamma_crit()
print(f"Critical value of gamma: {gamma_crit}")
