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
sub_trials = ["beta_E_1_0", "beta_E_0_1", "beta_E_0_01", "beta_E_0_001"]
sub_trials_latex = ["$t_{E} = 2.1\\times 10^{3}\\mathrm{s}$",
                    "$t_{E} = 2.1\\times 10^{4}\\mathrm{s}$",
                    "$t_{E} = 2.1\\times 10^{5}\\mathrm{s}$",
                    "$t_{E} = 2.1\\times 10^{6}\\mathrm{s}$"]
colours = ["forestgreen", "deepskyblue", "darkviolet", "firebrick"]
middle_paths = [f"{parent}/{trial}/{sub_trial}" for sub_trial in sub_trials]
param_files = [open(f"resources/{middle_path}/params.json") for middle_path in middle_paths]
param_dicts = [json.load(param_file) for param_file in param_files]

# Whether we'll plot on a fixed domain or not
fixed_domain = False
num_quants = 8

# Whether to save data or not
saving = [True] * num_quants

# The frequency of plotting
# num_lines = N_time / 1
num_lines = 50
base_sim = Simulation(param_dicts[0], middle_paths[0], fixed_domain, num_quants, saving, num_lines)

phi_f0, nu = base_sim.phi_f0_num, base_sim.nu_num
t_v, t_phi, t_E = base_sim.t_v_num, base_sim.t_phi_num, base_sim.t_E


def run_simulation(_params: dict, _middle_path: str):
    _sim = Simulation(_params, _middle_path, fixed_domain, num_quants, saving, num_lines)
    short_quants = ["phi_f", "E", "c"]
    _sim.prepare_figure(short_quants, 3, 1)
    _sim.solve()
    _sim.plot_traces()
    _sim.save_t_crit_vals()
    return _sim


E_min_arr = np.linspace(0.05, 0.7, 14)
sim_id = 0
if not read_csv:
    for i in range(3, len(sub_trials)):
        # Delete any existing t_crit files
        param_dict = param_dicts[i]
        middle_path = middle_paths[i]
        if os.path.exists(f"resources/{middle_path}/data/t_crit.csv"):
            os.remove(f"resources/{middle_path}/data/t_crit.csv")
        for E_min in E_min_arr:
            print(f"t_E: {t_E}, E_min: {E_min}")
            param_dict["phys"]["E_min"] = E_min
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
alpha_min_arr = ((E_min_arr * phi_f0 * (2 * (1 - nu) - phi_f0) * t_v)
                 / (2 * t_phi * (1 - phi_f0) * (1 + nu) * (1 - 2 * nu)))
t_crit_arr_analytic = np.log((1 / E_min_arr - 1) / (1 / alpha_min_arr - 1))

# Create the plots
fig, ax = plt.subplots(1, 1, figsize=(12, 3.333))
t_crit_arr_analytic = t_crit_arr_analytic[~np.isnan(t_crit_arr_analytic)]
t_crit_len = len(t_crit_arr_analytic)
for i in range(len(sub_trials)):
    ax.plot(E_min_arr[:t_crit_len], t_crit_numerical_list[i][:t_crit_len], color=colours[i], linestyle="solid",
            lw=2, label=sub_trials_latex[i])
ax.plot(E_min_arr[:t_crit_len], t_crit_arr_analytic, "--k", lw=2, label="Analytic")
ax.legend(loc="upper left")
ax.set_xlabel("$E_{\\mathrm{min}}$")
ax.set_ylabel("$t_{\\mathrm{crit}}$")
ax.set_yscale("log")
fig.savefig(f"resources/{middle_paths[0]}/plots/t_crit.png", bbox_inches="tight")
