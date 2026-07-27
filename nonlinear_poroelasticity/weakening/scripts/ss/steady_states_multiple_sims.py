"""
In this file, we compare the analytical steady states form multiple different simulations
by varying one of the parameters across different simulations. For example, we can
change the value of the fixed imposed fluid flux, Q_f.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import json

from sympy.solvers import solve
from sympy import Symbol

from steady_state import SteadyState

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 20})
plt.rcParams['text.usetex'] = True


# Whether we use pre-simulated data or want to generate new data
use_data = False

"""
Reading in our parameters and defining our paths
"""
parent = "phys"
trial = "porous_polymer"
sub_trial = "gamma_0_1_phi_f0_0_5"
dir_path = f"resources/{parent}/{trial}/{sub_trial}"
data_path = f"{dir_path}/data"
plot_path = f"{dir_path}/plots/ss_mult_sims"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)

# The parameters from the simulation
N_x = params["comp"]["N_x"]
# N_x = 1000

phi_f0 = params["ics"]["phi_f"]
nu = params["phys"]["nu"]
E_min = params["phys"]["E_min"]

# Whether we are in a fluid flux or pressure drop simulation
fluid_flux = "Q_f" in params
if not fluid_flux:
    Delta_p = params["bcs"]["Delta p"]

param_file.close()


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


# Varying the initial porosity, phi_f0
phi_f0_arr = np.linspace(0.0, 0.95, 96)

# Varying the effective pressure drop, gamma
gamma_min, gamma_max = 0.0, 0.5
gamma_arr = np.linspace(gamma_min, gamma_max, 101)

xi = np.linspace(0, 1, N_x + 1)


def steady_state_one_sim(_steady_state: SteadyState):
    """Finds properties of the steady state for a particular simulation.

    :return: Properties of the steady state in a tuple.
    """
    phi_f_ss, a_ss, B_ss = _steady_state.solve_analytic()
    sigma_xx_ss = calculate_sigma_xx(phi_f_ss, _steady_state.phi_f0,
                                     _steady_state.nu, _steady_state.E_min)
    dsigma_xx_ss_dx = np.gradient(sigma_xx_ss, _steady_state.xi)
    v_ss = _steady_state.Q_f
    # phi_f at x = 1, Delta P, -dpf_dx at x = 1 and a
    alpha_min = _steady_state.get_alpha_min()
    t_crit = np.log((1 / _steady_state.E_min - 1) / (1 / alpha_min - 1))
    return phi_f_ss[-1], a_ss, v_ss, alpha_min, t_crit


def phi_r_t_crit_one_sim(_steady_state: SteadyState):
    """Finds specific properties of the steady state (which do not require
    solving the entire system). This is to speed up computation time.

    :return: Properties of the steady state in a tuple.
    """
    phi_r = _steady_state.get_phi_r_pressure_drop()
    alpha_min = _steady_state.get_alpha_min()
    if alpha_min == 0:
        alpha_min = 1e-32
    t_crit = np.log((1 / _steady_state.E_min - 1) / (1 / alpha_min - 1))
    if t_crit < 0.001:
        t_crit = 0
    return phi_r, t_crit


params["ics"]["phi_f"] = 0.5
params["bcs"]["Delta p"] = 0.4
steady_state = SteadyState(params, xi)

# Setting up dict for later (each item will be a list of lists representing
# E_min and phi_f0 values)
ss_outputs_dict = {"phi_r": [], "t_crit": []}
subscripts = ["phi_r", "t_crit"]
N_gamma, N_phi_f0 = len(gamma_arr), len(phi_f0_arr)
if not use_data:
    for i in range(N_gamma):
        gamma = float(gamma_arr[i])
        # This inner dict represents sims for all phi_f0 values but a fixed value
        # of E_min or gamma
        inner_dict = {"phi_r": [], "t_crit": []}
        for j in range(N_phi_f0):
            phi_f0 = float(phi_f0_arr[j])
            print(f"Simulation ({i}, {j}): gamma = {round(gamma, 3)}, phi_f0 = {round(phi_f0, 3)}")
            # Find important steady state outputs
            params["ics"]["phi_f"] = phi_f0
            params["bcs"]["Delta p"] = gamma
            steady_state = SteadyState(params, xi)

            # Use the below line if we want to calculate full steady state. If we only
            # need t_crit and phi_r, use the line below that
            # ss_outputs = steady_state_one_sim(steady_state)
            ss_outputs = phi_r_t_crit_one_sim(steady_state)
            for k, output_key in enumerate(inner_dict.keys()):
                # Add each output to the correct category within the inner_dict
                inner_dict[output_key].append(ss_outputs[k])
        for k, output_key in enumerate(ss_outputs_dict.keys()):
            # Add each inner list to the full dict (has length N_phi_f0)
            ss_outputs_dict[output_key].append(inner_dict[output_key])
    for k, output_key in enumerate(ss_outputs_dict.keys()):
        # Convert each inner array into a numpy array of shape (N_E_min x N_phi_f0)
        ss_outputs_dict[output_key] = np.array(ss_outputs_dict[output_key])
else:
    for k, output_key in enumerate(ss_outputs_dict.keys()):
        Z_df = (pd.read_csv(f"{data_path}/ss_params_gamma_{subscripts[k]}.csv", index_col=0)
                .to_numpy().transpose())
        ss_outputs_dict[output_key] = Z_df


def alpha_min_1(_phi_f0: np.array, _nu: float, _Delta_p: float):
    """An expression E_min(phi_f0) to determine when alpha_min = 1.

    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _Delta_p: The pressure drop.
    :return: An array of E_min for which alpha_min = 1
    """
    numerator = 2 * _Delta_p * (1 - _phi_f0) * (1 + _nu) * (1 - 2 * _nu)
    denominator = _phi_f0 * (2 * (1 - _nu) - _phi_f0)
    return numerator / denominator


def case_1_case_2_bdry(_phi_f0: np.array, _nu: float, _E_min: float):
    """An expression gamma(phi_f0) to determine when alpha_min = 1.

    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _E_min: The minimum Young's modulus.
    :return: An array of gamma for which alpha_min = 1
    """
    numerator = _E_min * _phi_f0 * (2 * (1 - _nu) - _phi_f0)
    denominator = 2 * (1 - _phi_f0) * (1 + _nu) * (1 - 2 * _nu)
    return numerator / denominator


def case_2_case_3_bdry(_nu: float, _Delta_p: float):
    phi_f0_sym = Symbol('x')
    solns = solve(alpha_min_1(phi_f0_sym, _nu, _Delta_p) - 1, phi_f0_sym)
    return solns[0] if (0 < solns[0] < 1) else solns[1]


# E_min_mesh, phi_f0_mesh = np.meshgrid(E_min_arr, phi_f0_arr)
gamma_mesh, phi_f0_mesh = np.meshgrid(gamma_arr, phi_f0_arr)
latex_quants = ["$\\phi_{f,r}$", "$a$", "$v$", "$\\alpha_{\\mathrm{min}}$", "$t_{\\mathrm{crit}}$"]
cmap_names = ["viridis", "viridis", "viridis", "viridis", "viridis"]

# nfigs = 5
# # We will use the phi_r array to test the validity of the steady state.
# # If phi_r <= 0, then the steady state does not exist
# for k in range(nfigs):
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.333))
#     cmap_name = cmap_names[k]
#     output_key = list(ss_outputs_dict.keys())[k]
#     # Z is an array of shape (N_Q_f, N_phi_f0)
#     Z = ss_outputs_dict[output_key].transpose()
#     # Mask any unreasonable values (when alpha_min <= 1)
#     if output_key not in ["t_crit", "alpha_min"]:
#         Z[Z_alpha_min.transpose() <= 1] = np.nan
#     cmap = mpl.colormaps[cmap_name].copy()
#     cmap.set_bad(color="grey")
#     if output_key == "dpf_dx" or output_key == "t_crit" or output_key == "alpha_min":
#         CS = ax.pcolormesh(phi_f0_mesh, E_min_mesh, Z, cmap=cmap,
#                            norm=mpl.colors.LogNorm())
#     else:
#         CS = ax.pcolormesh(phi_f0_mesh, E_min_mesh, Z, cmap=cmap)
#     ax.set_xlabel("$\\phi_{f,0}$")
#     ax.set_ylabel("$E_{\\mathrm{min}} / E_0$")
#     cbar = fig.colorbar(CS)
#     cbar.ax.set_ylabel(latex_quants[k])
#     # Optional boundary curves
#     if output_key in ["alpha_min", "phi_r", "t_crit"] and not fluid_flux:
#         E_min_alpha_min_1 = alpha_min_1(phi_f0_arr, nu, Delta_p)
#         ax.plot(phi_f0_arr, E_min_alpha_min_1, "--k", label="$\\alpha_{\\mathrm{min}} = 1$")
#         ax.axvline(alpha_min_E_min(nu, Delta_p), linestyle=":", color="black",
#                    label="$\\alpha_{\\mathrm{min}} = E_{\\mathrm{min}}$")
#         ax.legend()
#         ax.set_ylim(0.05, 1.0)
#     fig.savefig(f"{plot_path}/_{subscripts[k]}.png", bbox_inches="tight",
#                 dpi=800)
#     Z_df = pd.DataFrame(Z)
#     fig.clf()
#     Z_df.to_csv(f"{data_path}/ss_params_{subscripts[k]}.csv")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
figsize = (6, 6.66)
fig = plt.figure(constrained_layout=True, figsize=figsize)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.2, 1], width_ratios=[1, 0.05])
ax = fig.add_subplot(gs[:, 0])
cmap_names = ["viridis", "Greys"]
latex_quants = ["$\\phi_{f,r}$", "$t_{\\mathrm{crit}}$"]
output_keys = ["phi_r", "t_crit"]
norms = [mpl.colors.Normalize(), mpl.colors.LogNorm(vmin=0.1, vmax=10)]
for k in range(len(cmap_names)):
    cmap = mpl.colormaps[cmap_names[k]].copy()
    cmap.set_bad(alpha=0)
    Z = ss_outputs_dict[output_keys[k]].transpose()
    pc = ax.pcolormesh(phi_f0_mesh, gamma_mesh, Z, cmap=cmap, norm=norms[k])
    ax.set_xlabel("$\\phi_{f,0}$")
    ax.set_ylabel("$\\gamma=\\Delta p / E_0$")
    cax = fig.add_subplot(gs[2 * k, 1])
    cbar = fig.colorbar(pc, cax=cax)
    cbar.ax.set_ylabel(latex_quants[k])
    gamma_case_1_case_2 = case_1_case_2_bdry(phi_f0_arr, nu, E_min)
    gamma_case_2_case_3 = case_1_case_2_bdry(phi_f0_arr, nu, 1)
    pd.DataFrame(Z).to_csv(f"{data_path}/ss_params_gamma_{subscripts[k]}.csv")
ax.plot(phi_f0_arr, gamma_case_1_case_2, "--r", label="$\\alpha_{\\mathrm{min}} = 1$",
             linewidth=1)
ax.plot(phi_f0_arr, gamma_case_2_case_3, "--r", label="$\\alpha_{\\mathrm{min}} = E_{\\mathrm{min}}$",
             linewidth=1)
ax.set_ylim(gamma_min, gamma_max)
fig.savefig(f"resources/paper_1/example_I/plots/qss_param_space.png", bbox_inches="tight", dpi=800)
