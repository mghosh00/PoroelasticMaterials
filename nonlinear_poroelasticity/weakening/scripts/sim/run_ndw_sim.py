"""
In this file we run the Simulation class from nondim_weakening.py
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

from nondim_weakening import Simulation

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
parent = "phys"
trial = "porous_polymer"
sub_trial = "Delta_p_0_1"
middle_path = f"{parent}/{trial}/{sub_trial}"
param_file = open(f"resources/{middle_path}/params.json")
params = json.load(param_file)

# Whether we'll plot on a fixed domain or not
plot_coord = "X"
num_quants = 8

# Whether to save data or not
saving = [True] * 4

# The frequency of plotting
# num_lines = N_time / 1
num_lines = 50

# Initialise simulation and figure
sim = Simulation(params, middle_path, plot_coord, num_quants, saving, num_lines)
short_quants = ["phi_f", "E", "c", "u_s"]
sim.prepare_figure(short_quants, 4, 1)

# Read in necessary data if using early-time similarity solution
ess_dict = {early_quant: pd.read_csv(f"resources/{middle_path}/data/ess_{early_quant}_xi.csv", index_col=0)
            .to_numpy()[:, 1] for early_quant in ["phi_f", "u_s"]}
responses_arr = pd.read_csv(f"resources/{middle_path}/data/ess_responses.csv", index_col=0).to_numpy()
ess_dict["a"], ess_dict["v"] = responses_arr[0, 0], responses_arr[0, 1]
sim.solve(ess_dict)
sim.plot_traces()
sim.plot_responses()
sim.plot_averages()
sim.save_responses_and_averages()
sim.save_t_crit_vals()
