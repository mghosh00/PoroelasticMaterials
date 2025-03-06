import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json
from PIL import Image

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters and defining our paths
"""
trial = "nondim_realistic_params"
sub_trial = "lower_bound"
dir_path = f"resources/{trial}/{sub_trial}"
data_path = f"{dir_path}/data"
plot_path = f"{dir_path}/plots"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)

# The parameters from the simulation
N_time = params["comp"]["N_time"]
delta_t = params["comp"]["delta_t"]

phi_f0 = params["ics"]["phi_f"]

L = params["phys"]["L"]
nu = params["phys"]["nu"]
mu = params["phys"]["mu"]

k_0 = params["scales"]["k"]
E_star = params["scales"]["E"]
v_star = params["scales"]["v"]

v_final = params["v"]["v_final"]

param_file.close()

t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star

epsilon_phi = t_v / t_phi


# Find the value of a(t) --- this comes from the u_s_array
u_s_array = pd.read_csv(f"{data_path}/u_s.csv").to_numpy()[1:, 2:]
a_array = u_s_array[0, :]

# Now, read the true phi_f array
phi_f_array = pd.read_csv(f"{data_path}/phi.csv").to_numpy()[1:, 2:]
phi_f_right = phi_f_array[-1, :]
xi_array = pd.read_csv(f"{data_path}/phi.csv").to_numpy()[1:, 1]

# For this workflow, we ASSUME that E is constant in time at x = 1. This
# will be true for the workflows we consider, and is approximately true
# when E does not change much (i.e. when t_E is very large compared to [t])
E_const = pd.read_csv(f"{data_path}/E.csv").to_numpy()[1, -1]


def dg_dphi(_phi_f: np.array, _phi_f0: float, _nu: float):
    """The derivative of the effective stress function.

    :param _phi_f: The porosity array.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :return: The effective stress.
    """
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    term1 = (1 - _phi_f0) ** 2 / (1 - _phi_f) ** 2
    term2 = 1 - 2 * _nu
    return (term1 + term2) / denominator


def k_e(_phi_f: np.array, _phi_f0: float):
    """The effective permeability function.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :return: The effective permeability.
    """
    numerator = (1 - _phi_f0) * _phi_f ** 2
    denominator = _phi_f0 ** 3 * (1 - _phi_f)
    return numerator / denominator


def dphi_dxi(_phi_f: np.array, _phi_f0: float, _nu: float, _a: np.array,
             _epsilon_phi: float, _v: float, _E_const: float):
    """Calculates the steady state value for E. Note that we must pass the true value
    for E at x = 1.

    :param _phi_f: The values of phi_f on the right for all time.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _a: The left boundary for phi_f for all time.
    :param _epsilon_phi: The ratio between the fluid flow and poroelastic timescales.
    :param _v: The phase-averaged velocity.
    :param _E_const: The constant value of the Young's modulus at x = 1.
    :return: The gradient dphi_dxi at the point x = 1.
    """
    _dg_dphi = dg_dphi(_phi_f, _phi_f0, _nu)
    _k_e = k_e(_phi_f, _phi_f0)
    numerator = _v * (1 - _phi_f) * (1 - _a)
    denominator = _epsilon_phi * _E_const * _phi_f * _k_e * _dg_dphi
    return - numerator / denominator


dphi_dxi_array = dphi_dxi(phi_f_right, phi_f0, nu, a_array, epsilon_phi,
                          v_final, float(E_const))

# Finally, we make an animation of phi_f and the linear approximation to phi_f
# at the right over time. These should match up if our condition is satisfied
# (that E is constant on the right)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)
whole_map = mpl.colormaps["Blues"]
cmap = mpl.colors.LinearSegmentedColormap.from_list(f"Blues_subset",
                                                    whole_map(np.linspace(0.3, 1.0, 100)))

# We store each frame separately before making the .gif file
if not os.path.exists(f"{plot_path}/frames"):
    os.makedirs(f"{plot_path}/frames")

images = []
for i in range(N_time):
    t = i * delta_t
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 20/3))

    # Plotting for phi
    ax.plot(xi_array, phi_f_array[:, i], color=cmap(norm(t)), label="True")
    phi_f_right_i = phi_f_right[i]
    phi_f_lin = phi_f_right_i + dphi_dxi_array[i] * (xi_array - 1)
    ax.plot(xi_array[-20:], phi_f_lin[-20:], "--r", label="Predicted gradient")
    ax.legend()
    ax.set_xlabel("$\\xi$")
    ax.set_ylabel("$\\phi_f$")
    ax.set_ylim(0, 1)

    # Saving the figure
    fig.savefig(f"{plot_path}/frames/frame_{i}.png", bbox_inches="tight")
    image = Image.open(f"{plot_path}/frames/frame_{i}.png")
    images.append(image)
    os.remove(f"{plot_path}/frames/frame_{i}.png")

# Make the .gif and delete the frames directory
images[0].save(f"{plot_path}/phi_f_gradient.gif", save_all=True,
               append_images=images[1:], duration=5, loop=0)
os.rmdir(f"{plot_path}/frames")
