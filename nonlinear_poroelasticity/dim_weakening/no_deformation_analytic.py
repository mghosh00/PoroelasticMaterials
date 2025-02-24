import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 30})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
trial = "analytic"
v_name = "v_0"
param_file = open(f"resources/{trial}/{v_name}/params.json")
params = json.load(param_file)

N_x = params['comp']['N_x']
N_time = params['comp']['N_time']
delta_t = params['comp']['delta_t']


# Setting up domain, parameters, and max index of sum
x_arr = np.linspace(0, 1, N_x + 1)
t_arr = np.linspace(0, N_time * delta_t, N_time)
alpha = 0.5
n_upper = 1000


def summand(_x_arr: np.array, _t: float, _alpha: float, _n: int):
    """Calculates the summand for the analytic solution to the solute concentration:

    \\frac{1}{(2n + 1)\\pi}\\sin((n + 1/2)\\pi x)e^{-\\alpha(n + 1/2)^{2}\\pi^{2}t}

    :param _x_arr: The array of the spatial coordinate.
    :param _t: The current timestep.
    :param _alpha: The diffusive parameter.
    :param _n: The entry in the sum.
    :return: The summand for the analytic solution for c.
    """
    sine = np.sin((_n + 1/2) * np.pi * _x_arr)
    exp = np.exp(-_alpha * (_n + 1/2) ** 2 * np.pi ** 2 * _t)
    denominator = (2 * _n + 1) * np.pi
    return sine * exp / denominator


# Using Quantity's plotting helper functions
c = Quantity("$c$", "Reds", 0)
c._mesh = x_arr
fig, ax = plt.subplots(1, 1)
c.set_ax(ax)
norm = mpl.colors.Normalize(vmin=0.0, vmax=float(t_arr[-1]))

# Calculating and plotting c at each time step
for t in t_arr:
    c_arr = 2 - 4 * sum([summand(x_arr, t, alpha, n) for n in range(n_upper)])
    c.f = c_arr
    c.plot(norm, t)

# Setting up the colorbar and labelling the plot
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c.cmap),
             orientation='vertical',
             label='$t$', ax=c.ax)
c.label_plot(x_label="$x$", title="")

# Saving the figure
fig.savefig(f"resources/{trial}/{v_name}/plots/analytic.png", bbox_inches="tight")
