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


meta_path = "paper_1/example_I"
_, _, meta_params = get_input_info(meta_path, "meta_info")
trial_params = meta_params["trial_params"]
quant_params = meta_params["quant_params"]
num_trials = trial_params["num_trials"]
num_trace_trials = 1 if "selected_subtrial" in quant_params else num_trials
ss_index = quant_params["selected_subtrial"] if num_trace_trials == 1 else -1
parents = [trial_params["parent"]] * num_trials
trials = [trial_params["trial"]] * num_trials
sub_trials = trial_params["sub_trials"]
sub_trials_trace = sub_trials if ss_index == -1 else [sub_trials[ss_index]]
labels = trial_params["labels"]
data_paths = []
data_paths_resp = []
plot_paths = []
param_sets = []
for i in range(num_trials):
    data_path_i, plot_path_i, param_set_i = get_input_info(f"{parents[i]}/{trials[i]}/{sub_trials[i]}")
    data_paths_resp.append(data_path_i)
    data_paths.append(data_path_i)
    plot_paths.append(plot_path_i)
    param_sets.append(param_set_i)

num_quants = quant_params["num_quants"]
short_quants = quant_params["short_quants"]
latex_quants = quant_params["latex_quants"]
colour_maps = quant_params["colour_maps"]
cmap_bounds = tuple(quant_params["cmap_bounds"]) if "cmap_bounds" in quant_params else (0.3, 1.0)
plot_coord = meta_params["plot_coord"]
plot_coord_tex = meta_params["plot_coord_tex"]
log_time = True if meta_params["log_time"] == "True" else False
tlabel = meta_params["tlabel"]

qss_str = "qss_" if "vertical" in trial_params else ""

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
data_paths_trace = data_paths if ss_index == -1 else [data_paths[ss_index]]
data_dicts = {i: {q: np.array(0) for q in short_quants} for i in range(num_trace_trials)}
for i, data_path in enumerate(data_paths_trace):
    for q in short_quants:
        fname = f"{data_path}/_{q}_{qss_str}{plot_coord}.csv"
        q_array = pd.read_csv(fname).to_numpy()[:, 2:]
        data_dicts[i][q] = q_array

# Recording the maximum and minimum values so that the plots are on the same scale
mins, maxes = [], []
for q in short_quants:
    q_arr_all = np.hstack([data_dicts[i][q] for i in range(num_trace_trials)])
    q_min, q_max = np.min(q_arr_all), np.max(q_arr_all)
    q_range = q_max - q_min
    mins.append(q_min - 0.04 * q_range)
    maxes.append(q_max + 0.04 * q_range)
mins, maxes = None, None
num_times = len(data_dicts[0][short_quants[0]][0, :])
inner_q_list = []
for j in range(num_quants):
    inner_q_list.append(Quantity(latex_quants[j], colour_maps[j], j,
                                 cmap_bounds=cmap_bounds))
    inner_q_list[j]._mesh = coord_arr

# Make copies of the quantities so that they correspond to different axes
quantities_all = [inner_q_list] * num_trace_trials

"""
Plotting
"""
if "horizontal" in trial_params:
    nrows, ncols = num_trace_trials, num_quants
    height_ratios = [1] * nrows
else:
    nrows, ncols = num_quants, num_trace_trials
    height_ratios = [1] * nrows
figsize = (quant_params["figsize"][0], quant_params["figsize"][1])
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                        sharex=True, gridspec_kw={'height_ratios': height_ratios})
if not isinstance(axs, plt.Axes):
    if ss_index != -1:
        axs = [[axs[j]] for j in range(num_quants)]
    else:
        axs = [axs]
else:
    axs = [[axs]]
plt.subplots_adjust(wspace=0.4)
t_start, t_end = float(times[0]), float(times[-1])
t_colorbar_init = 1e-12 if log_time else 0.0
if log_time:
    norm = mpl.colors.LogNorm(vmin=float(times[1]), vmax=t_end)
else:
    norm = mpl.colors.Normalize(vmin=0.0, vmax=t_end)

plot_freq = max(int(num_times / 20), 1)
for i in range(num_trace_trials):
    m_max = data_dicts[i][short_quants[0]].shape[1] - 1
    quantities = quantities_all[i]
    axs_list = [axs[j][i] for j in range(num_quants)]
    Quantity.set_axs(quantities, axs_list)
    for m in range(m_max + 1):
        t = float(times[int(m * len(times) / num_times)])
        for j, q in enumerate(short_quants):
            if plot_coord == "xi":
                quantities[j].f = data_dicts[i][q][:, m]
            else:
                quantities[j].plotting_f = data_dicts[i][q][:, m]
        if t == 0 and log_time:
            t = times[1]
        if m % plot_freq == 0 or m == m_max:
            Quantity.plot_quantities(quantities, norm, float(t), plot_coord=plot_coord)
    ylabel = "name" if (i == 0 or "vertical" in trial_params) else None
    Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex, ylabel=ylabel,
                            mins=mins, maxes=maxes, colourbar=False)
    if ss_index == -1 and "vertical" not in trial_params:
        quantities[0].ax.set_title(labels[i])

# Adding the colourbars
for j, q in enumerate(quantities_all[0]):
    if "vertical" in trial_params:
        divider = make_axes_locatable(axs[j][0])
        cax = divider.append_axes('top', size='5%', pad=0.4)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=q.cmap),
                     orientation='horizontal',
                     label=tlabel, cax=cax)
        cax.xaxis.set_ticks_position("top")
    else:
        if not "horizontal" in trial_params or j == num_quants - 1:
            divider = make_axes_locatable(axs[j][-1])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=q.cmap),
                         orientation='vertical',
                         label=tlabel, cax=cax)

# axs_list[0].set_ylim(-0.02, 0.52)
# Check plot directory exists
output_plot_path = f"resources/{meta_path}/plots"
if not os.path.exists(output_plot_path):
    os.makedirs(output_plot_path)
# Save figure
fig.savefig(f"{output_plot_path}/time_traces_{qss_str}{plot_coord}.png", bbox_inches="tight",
            dpi=600)

# Plotting E inset near x = 0 and t = 0
fig, ax = plt.subplots(1, 1, figsize=(3, 2))
m_min_sm, m_max_sm = 21, 32
t_start_sm, t_end_sm = float(times[m_min_sm * 13]), float(times[(m_max_sm + 1) * 13])

plot_freq = 2
E_small = Quantity("$E$", colour_maps[2], 2, cmap_bounds=cmap_bounds)
E_small._mesh = coord_arr
norm_small = mpl.colors.LogNorm(vmin=t_start_sm, vmax=t_end_sm)
Quantity.set_axs([E_small], [ax])
for m in range(m_min_sm-1, m_max_sm):
    t = float(times[int(m * len(times) / num_times)])
    E_small.plotting_f = data_dicts[0]["E"][:, m]
    if m % plot_freq == 0 or m == m_max_sm + 1:
        Quantity.plot_quantities([E_small], norm_small, float(t), plot_coord=plot_coord)
Quantity.annotate_plots([E_small], fig, norm_small, xlabel=None, ylabel=None,
                        mins=mins, maxes=maxes, colourbar=False)

# Adding the colourbar
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm_small, cmap=E_small.cmap),
             orientation='vertical',
             label=tlabel, cax=cax)
fig.savefig(f"{output_plot_path}/E_small_{qss_str}{plot_coord}.png", bbox_inches="tight",
            dpi=600)

# Extract timescales from the ss_index parameter set
scales = param_sets[ss_index]["scales"]
phys = param_sets[ss_index]["phys"]
bcs = param_sets[ss_index]["bcs"]
t_E = scales["t"]
t_phi = phys["mu"] * phys["L"] ** 2 / scales["k"] / scales["E"]
t_v = t_phi / bcs["Delta p"]

if not qss_str:
    """
    Plots of a, v and other time-dependent variables
    """
    response_params = meta_params["response_params"]
    avg_params = meta_params["avg_params"]
    var_names = response_params["var_names"]
    avg_names = avg_params["names"]
    num_vars = len(var_names)
    latex_vars = response_params["latex_var_names"]
    colours = response_params["colours"]
    yscales = response_params["yscales"]
    log_time_resp = response_params["log_time"] if "log_time" in response_params else log_time
    nrows_resp, ncols_resp, figsize_resp = response_params["nrows"], response_params["ncols"], tuple(response_params["figsize"])
    line_styles = trial_params["linestyles"]
    data_dicts = {i: {f: np.array(0) for f in var_names + avg_names} for i in range(num_trials)}
    times_list = []
    loc_list = ['upper left', 'upper right']
    ax_phi_y_min, ax_phi_y_max = 1, 0
    for i, data_path in enumerate(data_paths_resp):
        response_df = pd.read_csv(f"{data_path}/_responses.csv", index_col=0)
        times_list.append(response_df["Time"].to_numpy())
        for f in var_names:
            f_array = response_df[f]
            data_dicts[i][f] = f_array
        for avg in avg_names:
            avg_array = response_df[avg]
            data_dicts[i][avg] = avg_array
            if avg == "phi_f_bar" and i == ss_index:
                ax_phi_y_min = min(ax_phi_y_min, np.min(avg_array))
                ax_phi_y_max = max(ax_phi_y_max, np.max(avg_array))

    ax_phi_y_min -= 0.003
    ax_phi_y_max += 0.003

    fig_resp, axs_resp = plt.subplots(nrows=nrows_resp, ncols=ncols_resp, figsize=figsize_resp,
                                      sharex=True)
    plt.subplots_adjust(wspace=0.3)
    for j in range(num_vars):
        f, f_latex, colour, yscale, ax = var_names[j], latex_vars[j], colours[j], yscales[j], axs_resp[j]
        for i in range(num_trials):
            times, line_style, label = times_list[i], line_styles[i], labels[i]
            f_arr = data_dicts[i][f]
            if log_time_resp or f == "v":
                times, f_arr = times[1:], f_arr[1:]

            ax.plot(times, f_arr, color=colour, linestyle=line_style, label=label)
            ax.set_xlabel(tlabel)
            ax.set_ylabel(f_latex)
            if log_time_resp:
                ax.set_xscale("log")
            ax.set_yscale(yscale)
        ax.legend(fontsize='14', loc=loc_list[j])
        # Here we plot the locations of the different timescales
        times = times_list[0]
        ax.axvspan(0, t_phi / t_E, facecolor="black", alpha=0.5)
        ax.axvspan(t_phi / t_E, t_v / t_E, facecolor="black", alpha=0.3)
        ax.axvspan(t_v / t_E, times.max(), facecolor="black", alpha=0.1)

    # fig_resp.savefig(f"{output_plot_path}/responses.png", bbox_inches="tight",
    #                  dpi=600)

    """
    Plot of averages
    """

    figsize_avgs = tuple(avg_params["figsize"])
    colours_avg = avg_params["colours"]
    latex_labels_avg = avg_params["latex_labels"]
    # Create figure for various averages over time
    fig_avgs, ax_all = plt.subplots(figsize=figsize_avgs)
    phi_f_avg_arr, E_avg_arr, c_avg_arr = (data_dicts[ss_index][avg_names[0]],
                                           data_dicts[ss_index][avg_names[1]],
                                           data_dicts[ss_index][avg_names[2]])
    times = times_list[ss_index]
    i = 1 if log_time_resp else 0
    if "avg_and_resp" in meta_params:
        ax_all = axs_resp[-1]
    # Have a twin axis
    ax_phi = ax_all.twinx()

    # Here we plot the locations of the different timescales
    fill_por = ax_phi.axvspan(0, t_phi / t_E, facecolor="black", alpha=0.5, label="Poroelastic")
    fill_adv = ax_phi.axvspan(t_phi / t_E, t_v / t_E, facecolor="black", alpha=0.3, label="Advection")
    fill_weak = ax_phi.axvspan(t_v / t_E, times.max(), facecolor="black", alpha=0.1, label="Weakening")

    # Plot E_avg, c_avg and phi_f_avg over time on the same axis
    ln_phi = ax_phi.plot(times[i:], phi_f_avg_arr[i:], lw=2,
                         color=colours_avg[0], label=latex_labels_avg[0])
    ln_E = ax_all.plot(times[i:], E_avg_arr[i:], lw=2,
                       color=colours_avg[1], label=latex_labels_avg[1])
    ln_c = ax_all.plot(times[i:], c_avg_arr[i:], lw=2,
                       color=colours_avg[2], label=latex_labels_avg[2])
    ax_all.set_xlabel(tlabel)
    if log_time_resp:
        ax_all.set_xscale("log")
    lns = ln_phi + ln_c + ln_E
    labs = [ln.get_label() for ln in lns]
    fills = [fill_por, fill_adv, fill_weak]
    labs_fill = [f.get_label() for f in fills]

    leg1 = ax_phi.legend(lns, labs, loc="upper left", prop={"size": 14})
    ax_phi.add_artist(leg1)
    ax_phi.legend(fills, labs_fill, loc="center left", prop={"size": 14})
    ax_all.set_ylabel(latex_labels_avg[1] + ", " + latex_labels_avg[2])
    ax_phi.set_ylabel(latex_labels_avg[0])
    ax_phi.set_ylim(ax_phi_y_min, ax_phi_y_max)
    fig_resp.savefig(f"{output_plot_path}/responses.png", bbox_inches="tight",
                     dpi=600)
    # fig_avgs.savefig(f"{output_plot_path}/averages.png", bbox_inches="tight",
    #                  dpi=600)
