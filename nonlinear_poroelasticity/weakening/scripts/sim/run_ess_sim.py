"""
In this file we run the SimilaritySolution class from early_similarity_soln.py
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from early_similarity_soln import SimilaritySolution

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
parent = "phys"
trial = "porous_polymer"
sub_trial = "Delta_p_0_25_phif0_0_4"
middle_path = f"{parent}/{trial}/{sub_trial}"
param_file = open(f"resources/{middle_path}/params_eta.json")
params = json.load(param_file)

num_quants = 4

# Whether to save data or not
saving = [True] * 2

sim = SimilaritySolution(params, middle_path, saving, num_quants)
short_quants = ["f", "h"]
sim.prepare_figure(short_quants, 2, 1)
sim.solve()
sim.plot_traces()
sim.save_responses()
