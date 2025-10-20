"""
In this script, we create a diagram of the 1D poroelastic material and animate its
Young's modulus and solute concentration changing over time, along with the behaviour
of the left boundary.
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json
from PIL import Image

from fenics import Expression

from nonlinear_poroelasticity.weakening.scripts import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
parent = "phys"
trial = "porous_polymer"
sub_trial = "Delta_p_0_25"
param_file = open(f"resources/{parent}/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

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
E_min = params["phys"]["E_min"]

param_file.close()

"""
Read in quantity .csv files.
"""
short_quants = ["E", "c"]
data_path = f"resources/{parent}/{trial}/{sub_trial}/data"
plot_path = f"resources/{parent}/{trial}/{sub_trial}/plots"
file_names = [f"{data_path}/_{q}_x.csv" for q in short_quants]
data_dict = {}
for i, name in enumerate(short_quants):
    data_arr = pd.read_csv(file_names[i], header=None).to_numpy()
    times = np.array(data_arr[0, 2:], dtype=float)
    times[0] = 0.0
    coord_arr = np.array(data_arr[1:, 1], dtype=float)
    quant_arr_nan = np.array(data_arr[1:, 2:], dtype=float)
    data_dict[name] = quant_arr_nan

"""
Preparing the data.
"""
E = Quantity("$E$", "Greens", 1)
c = Quantity("$c$", "Purples", 2)

"""
Sorting out the timesteps.
"""
# t_tau = params["comp"]["t(tau)"] if "t(tau)" in params["comp"] else "tau"
# t_expr = Expression(t_tau, degree=1, tau=0.0, delta_tau=delta_tau, N_time=N_time)
# times = [float(t_expr(0.0))]
# for n in range(N_time):
#     t_expr.tau += delta_tau
#     times.append(float(t_expr(0.0)))
# times = np.array(times)

"""
The function for drawing a rectangle for E (creating the poroelastic material
bit by bit).
"""


def draw_rectangle(_ax: plt.Axes, x: float, _N_x: int, _E: float,
                   norm: mpl.colors.Normalize, colormap: mpl.colors.Colormap):
    """Draws a thin rectangle on the plot with colour determined by
    the value of E in a colormap.

    :param _ax: The axis to draw on.
    :param x: The x-coordinate of the left-hand side of the rectangle.
    :param _N_x: The number of x-coordinates in [0, 1].
    :param _E: The value of the Young's modulus at this point.
    :param norm: The colour norm.
    :param colormap: The string map used for colours.
    """
    colour = colormap(norm(_E))
    _ax.add_patch(plt.Rectangle((x, 0), 1 / _N_x, 1, color=colour))


def add_random_solute(_ax: plt.Axes, x: float, _N_x: int, _c: float, colour: str):
    """Adds solute at the specific x coordinate randomly distributed in the y
    direction of given density determined by _c.

    :param _ax: The axis to draw on.
    :param x: The x coordinate (in the centre of one of the thin rectangles).
    :param _N_x: The number of x-coordinates in [0, 1].
    :param _c: The concentration of solute at this coordinate.
    :param colour: The colour of each solute particle.
    """
    radius = 1 / (2 * N_x)
    num_particles = int(_c * 50)
    for j in range(num_particles):
        y = random.random()
        circle = plt.Circle((x, y), radius, facecolor=colour, alpha=0.3)
        _ax.add_patch(circle)


"""
Creating the frames.
"""

# We store each frame separately before making the .gif file
if not os.path.exists(f"{plot_path}/frames"):
    os.makedirs(f"{plot_path}/frames")

images = []
# The number of frames
num_frames = 51
# Setting up the colour gradient for E
E_norm = mpl.colors.Normalize(vmin=E_min, vmax=1.0)
for n in range(num_frames):
    m = int(n * len(times) / num_frames)
    """
    Set up figure for the overall plot
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.format_float_positional(float(times[m]), precision=3, unique=False,
                                   fractional=False, trim='k')
    print("Time:", t)

    # Get the arrays for the current timestep
    E_arr_nan = data_dict["E"][:, m]
    E_arr = E_arr_nan[~np.isnan(E_arr_nan)]
    c_arr_nan = data_dict["c"][:, m]
    c_arr = c_arr_nan[~np.isnan(c_arr_nan)]
    a = 1 - (len(E_arr) - 1) / N_x
    x_arr = np.linspace(a, 1, len(E_arr))

    for i in range(len(x_arr) - 1):
        edge, midpoint = float(x_arr[i]), float(x_arr[i] + 1 / (2 * N_x))
        # Add a thin rectangle with colour determined by E at each x point
        draw_rectangle(ax, edge, N_x, E_arr[i], E_norm, E.cmap)
        # Add a random distribution of solute determined by concentration at each x point
        add_random_solute(ax, midpoint, N_x, c_arr[i], c.cmap(E_norm(1)))

    # Label the plot and add a colourbar
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("$x$")
    fig.colorbar(mpl.cm.ScalarMappable(norm=E_norm, cmap=E.cmap),
                 orientation='vertical',
                 label="$E$", ax=ax)
    ax.set_title(f"$t = {t}$")

    # Save figure
    frame_path = f"{plot_path}/frames/frame_{n}.png"
    fig.savefig(frame_path, bbox_inches="tight")
    image = Image.open(frame_path)
    images.append(image)
    os.remove(frame_path)

# Make the .gif and delete the frames directory
images[0].save(f"{plot_path}/_diagram_animation.gif", save_all=True,
               append_images=images[1:], duration=250, loop=0)
os.rmdir(f"{plot_path}/frames")
