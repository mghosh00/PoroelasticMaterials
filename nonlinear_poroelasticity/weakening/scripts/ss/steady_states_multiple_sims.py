"""
In this file, we compare the analytical steady states form multiple different simulations
by varying one of the parameters across different simulations. For example, we can
change the value of the fixed imposed fluid flux, Q_f.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

from long_steady_state_analysis import get_phi_l, solve_analytic

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters and defining our paths
"""
trial = "long_steady_state"
sub_trial = "v_0_1"
dir_path = f"resources/{trial}/{sub_trial}"
data_path = f"{dir_path}/data"
plot_path = f"{dir_path}/plots/ss_mult_sims"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)

# The parameters from the simulation
N_x = params["comp"]["N_x"]
# N_x = 1000

phi_f0 = params["ics"]["phi_f"]

L = params["phys"]["L"]
nu = params["phys"]["nu"]
mu = params["phys"]["mu"]
E_min = params["phys"]["E_min"]
D_m = params["phys"]["D_m"]

k_0 = params["scales"]["k"]
E_star = params["scales"]["E"]
v_star = params["scales"]["v"]

c_left = params["bcs"]["c_left"]
sigma_l = params["bcs"]["sigma_left"]

param_file.close()

# Timescales
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
t_c = L ** 2 / D_m


def calculate_sigma_xx(_phi_f: np.array, _phi_f0: float, _nu: float, _E_min: float):
    """Calculates the steady state array for the Terzaghi stress tensor
    given the porosity array and other parameters.

    :param _phi_f: Porosity array.
    :param _phi_f0: Initial porosity.
    :param _nu: Poisson's ratio.
    :param _E_min: Minimal value of the Young's modulus.
    :return: The array for the Terzaghi stress tensor.
    """
    multiplier = _E_min / (2 * (1 - _phi_f0) * (1 + _nu) * (1 - 2 * _nu))
    term1 = (1 - _phi_f0) ** 2 / (1 - _phi_f)
    term2 = - 2 * _nu * (1 - _phi_f0)
    term3 = - (1 - 2 * _nu) * (1 - _phi_f)
    return multiplier * (term1 + term2 + term3)


# Varying the fluid flux, Q_f
Q_f_arr = np.linspace(0, 8, 101)
# Q_f = 1

# Varying the Poisson's ratio, nu
# nu_arr = np.linspace(-0.95, 0.45, 50)

# Varying the initial porosity, phi_f0
phi_f0_arr = np.linspace(0.05, 0.95, 91)

xi = np.linspace(0, 1, N_x + 1)


def steady_state_one_sim(_xi: np.array, _phi_f0: float, _nu: float,
                         _sigma_l: float, _E_min: float, _Q_f: float,
                         _t_phi: float, _t_v: float):
    """Finds properties of the steady state for a particular simulation.

    :param _xi: The spatial coordinate array.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _sigma_l: The left value of the stress.
    :param _E_min: The minimum Young's modulus.
    :param _Q_f: The fluid flux.
    :param _t_phi: The porosity timescale.
    :param _t_v: The advective timescale.
    :return: Properties of the steady state in a tuple.
    """
    phi_l = get_phi_l(_sigma_l, _E_min, _phi_f0, _nu)
    factor = (_t_phi * _Q_f * _phi_f0 ** 3) / (_t_v * _E_min * (1 - _phi_f0))
    phi_f_ss, a_ss, B_ss = solve_analytic(_xi, phi_l, _phi_f0, _nu, factor)
    sigma_xx_ss = calculate_sigma_xx(phi_f_ss, _phi_f0, _nu, _E_min)
    dsigma_xx_ss_dx = np.gradient(sigma_xx_ss, _xi)
    # phi_f at x = 1, Delta P, -dpf_dx at x = 1 and a
    return phi_f_ss[-1], -sigma_xx_ss[-1], -dsigma_xx_ss_dx[-1], a_ss


# Setting up dict for later (each item will be a list of lists representing
# Q_f and phi_f0 values)
ss_outputs_dict = {"phi_r": [], "Delta P": [], "dpf_dx": [], "a": []}
N_Q_f, N_phi_f0 = len(Q_f_arr), len(phi_f0_arr)
for i in range(N_Q_f):
    Q_f = float(Q_f_arr[i])
    # This inner dict represents sims for all phi_f0 values but a fixed value
    # of Q_f
    inner_dict = {"phi_r": [], "Delta P": [], "dpf_dx": [], "a": []}
    for j in range(N_phi_f0):
        phi_f0 = float(phi_f0_arr[j])
        print(f"Simulation ({i}, {j}): Q_f = {round(Q_f, 3)}, phi_f0 = {round(phi_f0, 3)}")
        # Find important steady state outputs
        ss_outputs = steady_state_one_sim(xi, phi_f0, nu, sigma_l,
                                          E_min, Q_f, t_phi, t_v)
        for k, output_key in enumerate(inner_dict.keys()):
            # Add each output to the correct category within the inner_dict
            inner_dict[output_key].append(ss_outputs[k])
    for k, output_key in enumerate(ss_outputs_dict.keys()):
        # Add each inner list to the full dict (has length N_phi_f0)
        ss_outputs_dict[output_key].append(inner_dict[output_key])
for k, output_key in enumerate(ss_outputs_dict.keys()):
    # Convert each inner array into a numpy array of shape (N_Q_f x N_phi_f0)
    ss_outputs_dict[output_key] = np.array(ss_outputs_dict[output_key])

# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 40/3), sharex=True)
# quant_arrays = [np.array(Delta_P_list), np.array(dpf_dx_list), np.array(a_list), np.array(phi_r_list)]
# latex_quants = ["$\\Delta P$", "$\\frac{dp_{f}}{dx}$", "$a$", "$\\phi_r$"]
# colours = ["orange", "firebrick", "darkviolet", "dodgerblue"]
#
# for i in range(len(axs)):
#     ax = axs[i]
#     ax.plot(phi_f0_arr, quant_arrays[i], color=colours[i])
#     ax.set_xlabel("$\\phi_{f,0}$")
#     ax.set_ylabel(latex_quants[i])
#
# fig.savefig(f"{plot_path}/ss_param_plot.png", bbox_inches="tight")


def Q_f_boundary_curve(_phi_f0: np.array, _nu: float, _E_min: float, _t_phi: float, _t_v: float):
    """An expression for Q_f(phi_f0) in the case where phi_r = 0. This determines
    where the porosity on the right is zero and hence is the limiting case.

    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _E_min: The minimal value of the Young's modulus.
    :param _t_phi: The poroelastic timescale.
    :param _t_v: The advective timescale.
    :return: Q_f as a function of phi_f0
    """
    multiplier = _E_min * _t_v / (2 * _t_phi * (1 + _nu) * (1 - 2 * _nu) * _phi_f0 ** 3)
    term1 = -(4 - 2 * _nu - 6 * _phi_f0 + 3 * _phi_f0 ** 2) * np.log(1 - _phi_f0)
    term2 = 3 * _nu * _phi_f0 ** 2
    term3 = (4 - 2 * _nu) / 3 * (-3 * _phi_f0 + 3 * _phi_f0 ** 2 - _phi_f0 ** 3)
    return multiplier * (term1 + term2 + term3)


Q_f_mesh, phi_f0_mesh = np.meshgrid(Q_f_arr, phi_f0_arr)
latex_quants = ["$\\phi_{r}$", "$\\Delta P$", "$-\\frac{dp_{f}}{dx}(1)$", "$a$"]
subscripts = ["phi_r", "DeltaP", "dpf_dx", "a"]
cmap_names = ["viridis", "viridis", "viridis", "viridis"]

nfigs = 4
# We will use the phi_r array to test the validity of the steady state.
# If phi_r <= 0, then the steady state does not exist
Z_phi_r = ss_outputs_dict["phi_r"].transpose()
for k in range(nfigs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    cmap = cmap_names[k]
    output_key = list(ss_outputs_dict.keys())[k]
    # Z is an array of shape (N_Q_f, N_phi_f0)
    Z = ss_outputs_dict[output_key].transpose()
    # Mask any unreasonable values (when phi_r <= 0)
    Z[Z_phi_r <= 1e-2] = np.ma.masked
    if output_key == "dpf_dx":
        CS = ax.contourf(phi_f0_mesh, Q_f_mesh, Z, 100, cmap=plt.cm.get_cmap(cmap),
                         norm=mpl.colors.LogNorm())
    else:
        CS = ax.contourf(phi_f0_mesh, Q_f_mesh, Z, 100, cmap=plt.cm.get_cmap(cmap))
    ax.plot(phi_f0_arr, Q_f_boundary_curve(phi_f0_arr, nu, E_min, t_phi, t_v),
            "--k", lw=1, label="Boundary curve")
    ax.legend()
    ax.set_xlabel("$\\phi_{f,0}$")
    ax.set_ylabel("$Q_f$")
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(latex_quants[k])
    fig.savefig(f"{plot_path}/_{subscripts[k]}.png", bbox_inches="tight")
