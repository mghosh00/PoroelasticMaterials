"""
In this script, we plot different simulations side by side.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import json

from fenics import Expression

from nonlinear_poroelasticity.weakening.scripts import Quantity

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters and defining our paths
"""


def get_input_info(files_dir: str, json_name: str = "params"):
    """Retrieves information about the input directories and parameters.

    :param files_dir: A string containing the location of the data.
    :param json_name: The name of the .json file in the directory.
    :return: A tuple containing the data path, plot path and parameter dict.
    """
    dir_path = f"resources/{files_dir}"
    data_path = f"{dir_path}/data"
    plot_path = f"{dir_path}/plots"
    param_file = open(f"{dir_path}/{json_name}.json")
    params = json.load(param_file)
    param_file.close()
    return data_path, plot_path, params


meta_path = "paper_1/example_III"
_, _, meta_params = get_input_info(meta_path, "meta_info")
num_trials = meta_params["trial_params"]["num_trials"]
parents = [meta_params["trial_params"]["parent"]] * num_trials
trials = [meta_params["trial_params"]["trial"]] * num_trials
sub_trials = meta_params["trial_params"]["sub_trials"]
labels = meta_params["trial_params"]["labels"]
data_paths = []
plot_paths = []
param_sets = []
for i in range(num_trials):
    data_path_i, plot_path_i, param_set_i = get_input_info(f"{parents[i]}/{trials[i]}/{sub_trials[i]}")
    data_paths.append(data_path_i)
    plot_paths.append(plot_path_i)
    param_sets.append(param_set_i)

num_quants = meta_params["quant_params"]["num_quants"]
short_quants = meta_params["quant_params"]["short_quants"]
latex_quants = meta_params["quant_params"]["latex_quants"]
colour_maps = meta_params["quant_params"]["colour_maps"]
fixed_domain = True if meta_params["fixed_domain"] == "True" else False
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"
log_time = True if meta_params["log_time"] == "True" else False
tlabel = meta_params["tlabel"]

"""
Computational parameters from the simulation
"""

# The parameters from the simulation
N_x = param_sets[0]["comp"]["N_x"]
coord_arr = np.linspace(0, 1, N_x + 1)

# Time-related parameters
delta_tau = param_sets[0]["comp"]["delta_tau"]
t_tau = param_sets[0]["comp"]["t(tau)"]
t_expr = Expression(t_tau, degree=1, tau=0.0, delta_tau=delta_tau)
times = [float(t_expr(0.0))]
for n in range(param_sets[0]["comp"]["N_time"]):
    t_expr.tau += delta_tau
    times.append(float(t_expr(0.0)))
times = np.array(times)

"""
Retrieving the data and making the quantity instances
"""

data_dicts = {i: {q: np.array(0) for q in short_quants} for i in range(num_trials)}
for i, data_path in enumerate(data_paths):
    for q in short_quants:
        q_array = pd.read_csv(f"{data_path}/_{q}_{plot_coord}.csv").to_numpy()[:, 2:]
        data_dicts[i][q] = q_array

# Recording the maximum and minimum values so that the plots are on the same scale
mins, maxes = [], []
for q in short_quants:
    q_arr_all = np.hstack([data_dicts[i][q] for i in range(num_trials)])
    q_min, q_max = np.min(q_arr_all), np.max(q_arr_all)
    q_range = q_max - q_min
    mins.append(q_min - 0.04 * q_range)
    maxes.append(q_max + 0.04 * q_range)
mins, maxes = None, None
num_times = len(data_dicts[0][short_quants[0]][0, :])
inner_q_list = []
for j in range(num_quants):
    inner_q_list.append(Quantity(latex_quants[j], colour_maps[j], j))
    inner_q_list[j]._mesh = coord_arr

# Make copies of the quantities so that they correspond to different axes
quantities_all = [inner_q_list] * num_trials

"""
Plotting
"""

nrows, ncols = num_quants, num_trials
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3),
                        sharex=True)
plt.subplots_adjust(wspace=0.3)
t_start, t_end = float(times[0]), float(times[-1])
if log_time:
    norm = mpl.colors.LogNorm(vmin=t_start, vmax=t_end)
else:
    norm = mpl.colors.Normalize(vmin=0.0, vmax=t_end)

for i in range(num_trials):
    quantities = quantities_all[i]
    Quantity.set_axs(quantities, [axs[j][i] for j in range(num_quants)])
    for m in range(num_times):
        t = float(times[int(m * len(times) / num_times)])
        for j, q in enumerate(short_quants):
            quantities[j].f = data_dicts[i][q][:, m]
        Quantity.plot_quantities(quantities, norm, t, fixed_domain=fixed_domain)
    print(quantities[0].f[-1])
    ylabel = "name" if i == 0 else None
    Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex, ylabel=ylabel,
                            mins=mins, maxes=maxes, colourbar=False)
    quantities[0].ax.set_title(labels[i])

# Adding the colourbars
for j, q in enumerate(quantities_all[0]):
    divider = make_axes_locatable(axs[j][-1])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=q.cmap),
                 orientation='vertical',
                 label=tlabel, cax=cax)

# Check plot directory exists
output_plot_path = f"resources/{meta_path}/plots"
if not os.path.exists(output_plot_path):
    os.makedirs(output_plot_path)
# Save figure
fig.savefig(f"{output_plot_path}/_time_traces_{plot_coord}.png", bbox_inches="tight")
