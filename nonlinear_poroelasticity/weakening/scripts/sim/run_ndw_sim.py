"""
In this file we run the Simulation class from nondim_weakening.py
"""

from fenics import *
import numpy as np
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
sub_trial = "Delta_p_0_25"
middle_path = f"{parent}/{trial}/{sub_trial}"
param_file = open(f"resources/{middle_path}/params.json")
params = json.load(param_file)

# Whether we'll plot on a fixed domain or not
fixed_domain = False
num_quants = 8

# Whether to save data or not
saving = [True] * num_quants

# Whether we use an early time solution to predict initial values of v or not
early_time_soln = False

# The frequency of plotting
# num_lines = N_time / 1
num_lines = 50

sim = Simulation(params, middle_path, fixed_domain, num_quants, saving, num_lines)
short_quants = ["phi_f", "E", "c"]
sim.prepare_figure(short_quants, 3, 1)
sim.solve()
sim.plot_traces()
sim.plot_responses()
sim.plot_averages()
sim.save_responses_and_averages()
sim.save_t_crit_vals()
