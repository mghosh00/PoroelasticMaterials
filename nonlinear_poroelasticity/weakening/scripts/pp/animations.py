import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import pandas as pd
import json
from PIL import Image
import time

from fenics import Expression

from nonlinear_poroelasticity.weakening.scripts import Quantity, SteadyState
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
trial = "long_steady_state"
sub_trial = "v_0_1"
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
delta_tau = params["comp"]["delta_tau"]

# Number of time steps
N_time = params["comp"]["N_time"]

# Number of mesh points
N_x = params["comp"]["N_x"]

"""
Physical parameters
"""
phi_f0 = params["ics"]["phi_f"]

L = params["phys"]["L"]
nu = params["phys"]["nu"]
mu = params["phys"]["mu"]
E_min = params["phys"]["E_min"]
D_m = params["phys"]["D_m"]

k_0 = params["scales"]["k"]
E_star = params["scales"]["E"]
v_star = params["scales"]["v"]

Q_f_final = params["Q_f"]["Q_f_final"]

c_left = params["bcs"]["c_left"]
sigma_l = params["bcs"]["sigma_left"]

param_file.close()

t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
t_c = L ** 2 / D_m

"""
Read in quantity .csv files.
"""
short_quants = ["phi", "E", "c", "sigma", "u_s", "v_s"]
data_path = f"resources/{trial}/{sub_trial}/data"
plot_path = f"resources/{trial}/{sub_trial}/plots"
file_names = [f"{data_path}/_{q}_{plot_coord}.csv" for q in short_quants]
data_dict = {}
for i, name in enumerate(short_quants):
    data_arr = pd.read_csv(file_names[i]).to_numpy()
    coord_arr = data_arr[:, 1]
    quant_arr_nan = data_arr[:, 2:]
    data_dict[name] = quant_arr_nan

# Find the analytical steady state for the porosity, Young's modulus and solute concentration
steady_state_phi = SteadyState(params, coord_arr)
phi_f_ss, a_ss, B_ss = steady_state_phi.solve_analytic()
E_ss = np.array([E_min] * (N_x + 1))
c_ss = np.array([c_left] * (N_x + 1))
ss_arrs = [phi_f_ss, E_ss, c_ss]

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
t_tau = params["comp"]["t(tau)"] if "t(tau)" in params["comp"] else "tau"
t_expr = Expression(t_tau, degree=1, tau=0.0, delta_tau=delta_tau, N_time=N_time)
times = [float(t_expr(0.0))]
for n in range(N_time):
    t_expr.tau += delta_tau
    times.append(float(t_expr(0.0)))
times = np.array(times)
print(times)

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
# The time step between each frame
tau_period = 0.002

# Set up the colorbars and label the plots
quant_names = ["phi", "E", "c"]
mins = np.array([np.min(data_dict[name]) for name in quant_names])
maxes = np.array([np.max(data_dict[name]) for name in quant_names])
ranges = maxes - mins
mins -= 0.1 * ranges
maxes += 0.1 * ranges
nrows, ncols = 3, 1
quantities = [quantities[name] for name in quant_names]

for n in range(int(N_time * delta_tau / tau_period)):
    m = n * int(tau_period / delta_tau)
    """
    Set up figure for the overall plot
    """
    # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
    # fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, (10 * nrows)/3), sharex=True)
    plt.subplots_adjust(wspace=1.0 * (ncols - 1))
    # axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
    axs_list = axs
    Quantity.set_axs(quantities, axs_list)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(times[-1]))
    t = np.round(float(times[m]), 3)
    print("Time:", t)

    # Set up the correct arrays for the timepoint
    for i, name in enumerate(quant_names):
        quantity = quantities[i]
        quantity_arr_n = data_dict[name][:, m]
        quantity.f = quantity_arr_n
        axs_list[i].plot(coord_arr, ss_arrs[i], "--k", label="Analytical steady state")
        axs_list[i].legend()

    # plot at the current timepoint
    lines = Quantity.plot_quantities(quantities, norm, t, fixed_domain=fixed_domain)

    Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex,
                            mins=mins, maxes=maxes,
                            titles=[f"$t={t:.3f}$"] + [None] * (len(quant_names) - 1))

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
        quantity_arr_n = data_dict[name][:, int(frame_no * tau_period / delta_tau)]
        quantity.f = quantity_arr_n
    lines = Quantity.plot_quantities(quantities, norm,
                                     times[frame_no], fixed_domain=fixed_domain)
    return tuple(lines)


# Create animation
# ani = animation.FuncAnimation(fig=fig, func=update,
#                               frames=int(N_time * delta_t / period), interval=5)

# Make the .gif and delete the frames directory
images[0].save(f"{plot_path}/all_animated.gif", save_all=True,
               append_images=images[1:], duration=120, loop=0)
os.rmdir(f"{plot_path}/frames")

# ani.save(filename=f"{plot_path}/phi_f_animated.gif", writer="pillow")
