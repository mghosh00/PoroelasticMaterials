"""
In this script, we animate the porosity solution from FEniCS (from nondim_weakening.py) and
compare it to two separate boundary layer predictions near x = 1, the first for early
times and the second for later times. The prediction for early times looks like:

\\phi_{f} = \\phi_{f,0} - v(1 - \\phi_{f,0})(2\\sqrt{t}{\\pi D_{1}}\\exp{(-\\frac{(1-x)^{2}}{4D_{1}t})}
                            - \\frac{1-x}{D_{1}}(1 - \\erf{(\\frac{1-x}{2\\sqrt{D_{1}t}})})),

where \\phi_{f,0} is the initial porosity, v is the nondimensional flux and D_{1} is a diffusive
coefficient for the BL equation.

For late times, the boundary layer in x produces the following time-independent prediction:

\\phi_{f} = (\\frac{v(1-x)}{D_{2}})^{\\frac{1}{4}}.

We will compare the result from the numerics to these two boundary layer predictions.
"""


import os
import numpy as np
import scipy.special as ss
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

# Whether we produce a .gif or set of traces
gif = True
# Whether we plot on log-log axes or not
log = False
log_text = "_log" if log else ""

# Whether we'll plot on a fixed domain or not
fixed_domain = True
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"
plot_coord_tex = f"log$(1 - ${plot_coord_tex})" if log else plot_coord_tex


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
Define model parameters
"""

# Length of domain, L
L = params["phys"]["L"]

# Initial porosity, \\phi_{f,0}
phi_f0 = params["ics"]["phi_f"]

# Poisson ratio and viscosity
nu = params["phys"]["nu"]
mu = params["phys"]["mu"]

# Permeability scale
k_0 = params["scales"]["k"]

# Solute concentration, Young's modulus and velocity scales
E_star = params["scales"]["E"]
v_star = params["scales"]["v"]
v = params["v"]["v_final"]

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star

# The collective "diffusion coefficient" for the early equation
D_phi_early = (1 * (1 - nu) * t_v) / (t_phi * (1 + nu) * (1 - 2 * nu))

# The collective "diffusion coefficient" for the late equation
D_phi_late = ((1 * (1 - phi_f0) * ((1 - phi_f0) ** 2 + 1 - 2 * nu) * t_v) /
              (8 * t_phi * (1 + nu) * (1 - 2 * nu) * phi_f0 ** 3))

"""
Read in quantity .csv files.
"""
short_quants = ["phi"]
latex_quants = ["log$(\\phi_{f})$" if log else "$\\phi_{f}$"]
colours = ["blue"]
data_path = f"resources/{trial}/{sub_trial}/data"
plot_path = f"resources/{trial}/{sub_trial}/plots"
file_names = [f"{data_path}/{q}_{plot_coord}.csv" for q in short_quants]
data_dict = {}
for i, name in enumerate(short_quants):
    data_arr = pd.read_csv(file_names[i]).to_numpy()
    coord_arr = data_arr[:, 1]
    quant_arr_nan = data_arr[:, 2:]
    data_dict[name] = quant_arr_nan

times = np.linspace(0, N_time * delta_t, N_time + 1)
coord_arr[0] = 0
coord_arr[-1] = 1

# if log:
#     coord_arr = 1 - coord_arr


"""
Defining the two boundary layer solutions
"""


def phi_f_bl_early(_x: np.array, _t: float, _phi_f0: float, _v: float,
                   _D_phi_early: float):
    """Returns the np.array solution in the early boundary layer. This is
    described at the top of the file.

    :param _x: The spatial coordinate array.
    :param _t: The specific timepoint.
    :param _phi_f0: The initial porosity.
    :param _v: The nondimensional volume flux.
    :param _D_phi_early: The diffusion coefficient for the problem.
    :return: The porosity array at this timepoint within the early BL.
    """
    term2 = 2 * np.sqrt(_t / (np.pi * _D_phi_early)) * np.exp(- (1 - _x) ** 2 / (4 * _D_phi_early * _t))
    term3 = - (1 - _x) / _D_phi_early * (1 - ss.erf((1 - _x) / (2 * np.sqrt(_D_phi_early * _t))))
    return _phi_f0 - _v * (1 - _phi_f0) * (term2 + term3)


def phi_f_bl_late(_x: np.array, _v: float, _D_phi_late: float):
    """Returns the np.array solution in the late boundary layer. This is
    described at the top of the file.

    :param _x: The spatial coordinate array.
    :param _v: The nondimensional volume flux.
    :param _D_phi_late: The diffusion coefficient for the problem.
    :return: The porosity array at this timepoint within the late BL.
    """
    return (_v * (1 - _x) / _D_phi_late) ** (1 / 4)


"""
Loop over time steps and plot each frame
"""

# We store each frame separately before making the .gif file
if not os.path.exists(f"{plot_path}/frames"):
    os.makedirs(f"{plot_path}/frames")


images = []
period = 0.0004

# Set up the colorbars and label the plots
mins = np.array([np.nanmin(data_dict[name]) for name in short_quants])
maxes = np.array([np.nanmax(data_dict[name]) for name in short_quants])

ranges = maxes - mins
if log:
    xmin, xmax = np.log(np.min(coord_arr)), np.log(np.max(coord_arr))
    mins, maxes = np.log(mins), np.log(maxes)
    print(mins, maxes)
else:
    mins -= 0.1 * ranges
    maxes += 0.1 * ranges
    xmin, xmax = coord_arr[0], coord_arr[-1]
    xmin -= 0.1 * (coord_arr[-1] - coord_arr[0])
    xmax += 0.1 * (coord_arr[-1] - coord_arr[0])
nrows, ncols = 1, 1

if not gif:
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, (10 * nrows)/3), sharex=True)
    plt.subplots_adjust(wspace=1.0 * (ncols - 1))
    axs_list = [axs]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)
    phi_f = Quantity(latex_quants[0], "Blues", 0)
    phi_f.mesh_fixed = np.log(1 - coord_arr) if log else coord_arr
    phi_f.set_ax(axs_list[0])

early_label = "Early boundary layer"
late_label = "Late boundary layer"
fenics_label = "FEniCS solution"
n_end = int(N_time * delta_t / period) + 1

for n in range(n_end):
    if n != n_end - 1:
        m = n * int(period / delta_t)
        t = m * delta_t
    else:
        m = -1
        t = N_time * delta_t
    N_x_current = np.count_nonzero(~np.isnan(data_dict["phi"][:, m]))
    # print(N_x_current)
    i_min, i_max = (N_x + 1 - N_x_current, N_x + 1)
    coord_arr_n = coord_arr[i_min:i_max]
    # print(coord_arr_n)

    # Calculate quantities within boundary layer
    early_bl_quants = []
    late_bl_quants = []
    phi_f_early = phi_f_bl_early(coord_arr_n, t, phi_f0, v, D_phi_early)
    phi_f_late = phi_f_bl_late(coord_arr_n, v, D_phi_late)
    early_bl_quants.append(phi_f_early)
    late_bl_quants.append(phi_f_late)

    """
    Set up figure for the overall plot
    """
    if gif:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, (10 * nrows)/3), sharex=True)
        plt.subplots_adjust(wspace=1.0 * (ncols - 1))
        axs_list = [axs]

    print("Time:", round(t, 3))

    # Set up the correct arrays for the timepoint
    for i, name in enumerate(short_quants):
        ax = axs_list[i]
        coord_arr_n = np.log(1 - coord_arr_n) if log else coord_arr_n
        early_bl_n = np.log(early_bl_quants[i]) if log else early_bl_quants[i]
        late_bl_n = np.log(late_bl_quants[i]) if log else late_bl_quants[i]
        fenics_arr_n = np.log(data_dict[name][:, m]) if log else data_dict[name][:, m]
        if gif:
            line_early_bl = ax.plot(coord_arr_n, early_bl_n, "--g", label=early_label)
            line_late_bl = ax.plot(coord_arr_n, late_bl_n, "--r", label=late_label)
            line_fenics = ax.plot(coord_arr_n, fenics_arr_n[i_min:i_max], color=colours[i], label=fenics_label)
            ax.set_xlabel(plot_coord_tex)
            ax.set_ylabel(latex_quants[i])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(mins[i], maxes[i])
            ax.set_title(f"Time = {round(t, 3)}")
        else:
            phi_f.f_fixed = fenics_arr_n
            # line_early_bl = ax.plot(coord_arr_n, early_bl_n, "--r",
            #                         label=early_label if n == 0 else None)
            line_late_bl = ax.plot(coord_arr_n, late_bl_n, "--r",
                                   label=late_label if n == 0 else None)
            line_fenics = phi_f.plot(norm, t, fixed_domain=True,
                                     label=fenics_label if n == 0 else None)
        ax.legend()
        # if log:
        #     ax.set_xscale("log")
        #     ax.set_yscale("log")

    if gif:
        # Save figure
        frame_path = f"{plot_path}/frames/frame_{n}.png"
        fig.savefig(frame_path, bbox_inches="tight")
        image = Image.open(frame_path)
        images.append(image)
        os.remove(frame_path)

if gif:
    # Make the .gif and delete the frames directory
    images[0].save(f"{plot_path}/{short_quants[0]}_animated_bl{log_text}.gif", save_all=True,
                   append_images=images[1:], duration=100, loop=0)
    os.rmdir(f"{plot_path}/frames")
else:
    # Set up the colorbar and save the figure
    Quantity.annotate_plots([phi_f], fig, norm, plot_coord_tex,
                            mins=[-2], maxes=maxes)
    fig.savefig(f"{plot_path}/{short_quants[0]}_bl_traces{log_text}.png", bbox_inches="tight")
