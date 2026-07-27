"""
In this file we run multiple simulations and record their t_crit values (when
the pores close on the right, if ever). We vary parameter values and plot the
actual t_crit values against the predicted ones from the asymptotics.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

from nondim_weakening import Simulation

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

# Whether we read in from t_crit.csv or not
read_csv = True

"""
Reading in our parameters
"""
parent = "phys"
trial = "enzymatic"
sub_trials = ["c_10_minus_5", "c_10_minus_4", "c_10_minus_3", "c_10_minus_2"]
sub_trials_latex = ["$\\epsilon = 0.038$",
                    "$\\epsilon = 0.38$",
                    "$\\epsilon = 3.8$",
                    "$\\epsilon = 38$"]
linestyles = ["solid", "dashed", "dashdot", "dotted"]
middle_paths = [f"{parent}/{trial}/{sub_trial}" for sub_trial in sub_trials]
param_files = [open(f"resources/{middle_path}/params.json") for middle_path in middle_paths]
param_dicts = [json.load(param_file) for param_file in param_files]

# Whether we'll plot on a fixed domain or not
plot_coord = "X"
num_quants = 8

# Whether to save data or not
saving = [False] * num_quants

# The frequency of plotting
# num_lines = N_time / 1
num_lines = 50
base_sim = Simulation(param_dicts[0], middle_paths[0], plot_coord, num_quants, saving, num_lines)

phi_f0, nu = base_sim.phi_f0_num, base_sim.nu_num
E_min, t_E = base_sim.E_min_num, base_sim.t_E_num


def run_simulation(_params: dict, _middle_path: str):
    _sim = Simulation(_params, _middle_path, plot_coord, num_quants, saving, num_lines)
    short_quants = ["phi_f", "E", "c"]
    _sim.prepare_figure(short_quants, 3, 1)
    _sim.solve()
    _sim.save_t_crit_vals()
    return _sim


t_E_list = [param_dict["scales"]["t"] for param_dict in param_dicts]
params = param_dicts[0]
t_phi = params["phys"]["mu"] * params["phys"]["L"] ** 2 / params["scales"]["k"] / params["scales"]["E"]
epsilon_list = [t_phi / t_E for t_E in t_E_list]
gamma_arr = np.linspace(0.87, 0.6, 10)
sim_id = 0
if not read_csv:
    for i in range(len(sub_trials)):
        # Delete any existing t_crit files
        param_dict = param_dicts[i]
        t_E = param_dict["scales"]["t"]
        middle_path = middle_paths[i]
        if os.path.exists(f"resources/{middle_path}/data/t_crit.csv"):
            os.remove(f"resources/{middle_path}/data/t_crit.csv")
        for gamma in gamma_arr:
            print(f"t_E: {t_E}, gamma: {gamma}")
            param_dict["bcs"]["Delta p"] = gamma
            param_dict["sim_id"] = sim_id
            sim = run_simulation(param_dict, middle_path)
            sim_id += 1
            if sim.t_crit >= sim.N_time * sim.delta_tau:
                break

t_crit_numerical_list = []
for i in range(len(sub_trials)):
    t_crit_df = pd.read_csv(f"resources/{middle_paths[i]}/data/t_crit.csv", index_col=0)
    t_crit_arr_numerical = t_crit_df["t_crit"].to_numpy()
    t_crit_numerical_list.append(t_crit_arr_numerical)

# Analytical prediction of t_crit
alpha_min_arr = ((E_min * phi_f0 * (2 * (1 - nu) - phi_f0))
                 / (2 * gamma_arr * (1 - phi_f0) * (1 + nu) * (1 - 2 * nu)))
t_crit_arr_analytic = np.log((1 / E_min - 1) / (1 / alpha_min_arr - 1))

# Create the plots
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax_nd, ax_d = axs[0], axs[1]
t_crit_arr_analytic = t_crit_arr_analytic[~np.isnan(t_crit_arr_analytic)]
t_crit_arr_analytic = t_crit_arr_analytic[1:]
t_crit_len = len(t_crit_arr_analytic) + 1
for i in range(len(sub_trials)):
    ax_nd.plot(gamma_arr[1:t_crit_len], t_crit_numerical_list[i][1:t_crit_len], color="deeppink", linestyle=linestyles[i],
            lw=2, label=sub_trials_latex[i])
ax_nd.plot(gamma_arr[1:t_crit_len], t_crit_arr_analytic, color="black", linestyle="solid",
        lw=2, label="Analytic")
ax_nd.set_ylabel("$t_{\\mathrm{crit}}$")
for i in range(len(sub_trials)):
    ax_d.plot(gamma_arr[1:t_crit_len], t_crit_numerical_list[i][1:t_crit_len] / epsilon_list[i], color="deeppink", linestyle=linestyles[i],
            lw=2, label=sub_trials_latex[i])
ax_d.set_ylabel("$t_{\\mathrm{crit}}/\\epsilon$")
ax_nd.legend(loc="upper right", prop={"size": 15})
ax_nd.set_xlabel("$\\gamma$")
ax_d.set_xlabel("$\\gamma$")
ax_nd.set_yscale("log")
ax_d.set_yscale("log")
fig.savefig(f"resources/paper_1/example_II/plots/t_crit_combined.png", bbox_inches="tight",
            dpi=500)
