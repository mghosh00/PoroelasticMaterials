import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import pandas as pd
import json
from PIL import Image
import time

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
trial = "nondim_realistic_params"
sub_trial = "lower_bound"
param_file = open(f"resources/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

# Whether we'll plot on a fixed domain or not
fixed_domain = False
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"
num_quants = 6

# Whether to save data or not
saving = [False] * num_quants

"""
Computational parameters
"""

# Size of time step
delta_t = params["comp"]["delta_t"]

# Number of time steps
N_time = params["comp"]["N_time"]

# Number of mesh points
N_x = params["comp"]["N_x"]

"""
Read in quantity .csv files.
"""
short_quants = ["phi", "E", "c", "sigma", "u_s", "v_s"]
data_path = f"resources/{trial}/{sub_trial}/data"
plot_path = f"resources/{trial}/{sub_trial}/plots"
file_names = [f"{data_path}/{q}_{plot_coord}.csv" for q in short_quants]
data_dict = {}
for i, name in enumerate(short_quants):
    data_arr = pd.read_csv(file_names[i]).to_numpy()
    coord_arr = data_arr[:, 1]
    quant_arr_nan = data_arr[:, 2:]
    data_dict[name] = quant_arr_nan

"""
Define the solutions phi_f, E and c
"""

phi_f = Quantity("$\\phi_{f}$", "Blues", 0)
E = Quantity("$E$", "Purples", 1)
c = Quantity("$c$", "Reds", 2)
sigma = Quantity("$\\sigma_{xx}'$", "Greys", 3)
u_s = Quantity("$u_s$", "Greens", 4)
v_s = Quantity("$v_{s}$", "YlOrBr", 5)


quantities = {"phi": phi_f, "E": E, "c": c, "sigma": sigma, "u_s": u_s, "v_s": v_s}
times = np.linspace(0, N_time * delta_t, N_time + 1)

# Setting up the mesh
for quantity in quantities.values():
    if fixed_domain:
        quantity.mesh_fixed = coord_arr
    else:
        quantity._mesh = coord_arr


"""
Loop over time steps and plot each frame
"""

# We store each frame separately before making the .gif file
if not os.path.exists(f"{plot_path}/frames"):
    os.makedirs(f"{plot_path}/frames")


images = []
period = 0.0005

# Set up the colorbars and label the plots
quant_names = ["phi"]
mins = np.array([np.min(data_dict[name]) for name in quant_names])
maxes = np.array([np.max(data_dict[name]) for name in quant_names])
ranges = maxes - mins
mins -= 0.1 * ranges
maxes += 0.1 * ranges
nrows, ncols = 1, 1
quantities = [quantities[name] for name in quant_names]

for n in range(int(N_time * delta_t / period)):
    m = n * int(period / delta_t)
    """
    Set up figure for the overall plot
    """
    # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
    # fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
    plt.subplots_adjust(wspace=1.0 * (ncols - 1))
    # axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
    axs_list = [axs]
    Quantity.set_axs(quantities, axs_list)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

    print("Time:", np.round(n * period, 3))

    # Set up the correct arrays for the timepoint
    for i, name in enumerate(quant_names):
        quantity = quantities[i]
        quantity_arr_n = data_dict[name][:, m]
        quantity.f = quantity_arr_n

    # plot at the current timepoint
    lines = Quantity.plot_quantities(quantities, norm, m * delta_t, fixed_domain=fixed_domain)

    Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex,
                            mins=mins, maxes=maxes)

    # Save figure
    frame_path = f"{plot_path}/frames/frame_{n}.png"
    fig.savefig(frame_path, bbox_inches="tight")
    image = Image.open(frame_path)
    images.append(image)
    os.remove(frame_path)

"""
Set up figure for the overall plot
"""
# nrows, ncols = 3, 2
# # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
# # fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
# fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
# plt.subplots_adjust(wspace=1.0 * (ncols - 1))
# axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
# Quantity.set_axs(quantities, axs_list)
# norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)
#
# # Set up the colorbars and label the plots
# mins = [np.min(data_list[i]) for i in range(len(quantities))]
# maxes = [np.max(data_list[i]) for i in range(len(quantities))]
# Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex,
#                         mins=mins, maxes=maxes)
#
# # Set up the correct arrays for the timepoint
# for i in range(len(quantities)):
#     quantity = quantities[i]
#     quantity_arr_n = data_list[i][:, 0]
#     quantity.f = quantity_arr_n
#
# # plot at the initial timepoint
# lines = Quantity.plot_quantities(quantities, norm, 0, fixed_domain=fixed_domain)


def update(frame_no: int):
    # fig.clf()
    for i, name in enumerate(quant_names):
        quantity_arr_n = data_dict[name][:, int(frame_no * period / delta_t)]
        quantity.f = quantity_arr_n
    lines = Quantity.plot_quantities(quantities, norm,
                                     frame_no * delta_t, fixed_domain=fixed_domain)
    return tuple(lines)


# Create animation
# ani = animation.FuncAnimation(fig=fig, func=update,
#                               frames=int(N_time * delta_t / period), interval=5)

# Make the .gif and delete the frames directory
images[0].save(f"{plot_path}/{quant_names[0]}_animated.gif", save_all=True,
               append_images=images[1:], duration=100, loop=0)
os.rmdir(f"{plot_path}/frames")

# ani.save(filename=f"{plot_path}/phi_f_animated.gif", writer="pillow")
