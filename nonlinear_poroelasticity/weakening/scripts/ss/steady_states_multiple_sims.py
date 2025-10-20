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
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True


# Whether we use pre-simulated data or want to generate new data
use_data = False

"""
Reading in our parameters and defining our paths
"""
parent = "phys"
trial = "enzymatic"
sub_trial = "beta_E_1_0"
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


# Varying the fluid flux, Q_f
# Q_f_arr = np.linspace(0, 8, 101)
# Q_f = 1

# Varying the minimal Young's modulus, E_min
E_min_arr = np.linspace(0.05, 1.0, 96)

# Varying the Poisson's ratio, nu
# nu_arr = np.linspace(-0.95, 0.45, 50)

# Varying the initial porosity, phi_f0
phi_f0_arr = np.linspace(0.05, 0.95, 91)

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


# Setting up dict for later (each item will be a list of lists representing
# E_min and phi_f0 values)
ss_outputs_dict = {"phi_r": [], "a": [], "v": [], "alpha_min": [], "t_crit": []}
subscripts = ["phi_r", "a", "v", "alpha_min", "t_crit"]
N_E_min, N_phi_f0 = len(E_min_arr), len(phi_f0_arr)
if not use_data:
    for i in range(N_E_min):
        E_min = float(E_min_arr[i])
        # This inner dict represents sims for all phi_f0 values but a fixed value
        # of E_min
        inner_dict = {"phi_r": [], "a": [], "v": [], "alpha_min": [], "t_crit": []}
        for j in range(N_phi_f0):
            phi_f0 = float(phi_f0_arr[j])
            print(f"Simulation ({i}, {j}): E_min = {round(E_min, 3)}, phi_f0 = {round(phi_f0, 3)}")
            # Find important steady state outputs
            params["ics"]["phi_f"] = phi_f0
            params["phys"]["E_min"] = E_min
            steady_state = SteadyState(params, xi)
            ss_outputs = steady_state_one_sim(steady_state)
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
        Z_df = (pd.read_csv(f"{data_path}/ss_params_{subscripts[k]}.csv", index_col=0)
                .to_numpy().transpose())
        ss_outputs_dict[output_key] = Z_df


nrows, ncols = 2, 2
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 20/3), sharex=True)
axs = [axs[i][j] for i in range(nrows) for j in range(ncols)]
quant_arrays = [ss_outputs_dict["a"][:, 45], ss_outputs_dict["v"][:, 45],
                ss_outputs_dict["alpha_min"][:, 45], ss_outputs_dict["t_crit"][:, 45]]
latex_quants = ["$a$", "$v$", "$\\alpha_{\\mathrm{min}}$", "$t_{\\mathrm{crit}}$"]
colours = ["crimson", "firebrick", "darkviolet", "dodgerblue"]

Z_phi_r = ss_outputs_dict["phi_r"].transpose()
Z_alpha_min = ss_outputs_dict["alpha_min"][:, :]
for i in range(len(axs)):
    ax = axs[i]
    arr = quant_arrays[i]
    # arr[Z_alpha_min[:, 45] <= 1] = np.nan
    ax.plot(E_min_arr, arr, color=colours[i])
    ax.set_xlabel("$E_{\\mathrm{min}}$")
    ax.set_ylabel(latex_quants[i])
    ax.set_xlim(0.0, 1.0)

fig.savefig(f"{plot_path}/ss_param_plot.png", bbox_inches="tight")


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


def alpha_min_E_min(_nu: float, _Delta_p: float):
    phi_f0_sym = Symbol('x')
    solns = solve(alpha_min_1(phi_f0_sym, _nu, _Delta_p) - 1, phi_f0_sym)
    return solns[0] if (0 < solns[0] < 1) else solns[1]


E_min_mesh, phi_f0_mesh = np.meshgrid(E_min_arr, phi_f0_arr)
latex_quants = ["$\\phi_{f,r}$", "$a$", "$v$", "$\\alpha_{\\mathrm{min}}$", "$t_{\\mathrm{crit}}$"]
cmap_names = ["viridis", "viridis", "viridis", "viridis", "viridis"]

nfigs = 5
# We will use the phi_r array to test the validity of the steady state.
# If phi_r <= 0, then the steady state does not exist
for k in range(nfigs):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    cmap_name = cmap_names[k]
    output_key = list(ss_outputs_dict.keys())[k]
    # Z is an array of shape (N_Q_f, N_phi_f0)
    Z = ss_outputs_dict[output_key].transpose()
    # Mask any unreasonable values (when alpha_min <= 1)
    if output_key not in ["t_crit", "alpha_min"]:
        Z[Z_alpha_min.transpose() <= 1] = np.nan
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(color="grey")
    if output_key == "dpf_dx" or output_key == "t_crit" or output_key == "alpha_min":
        CS = ax.pcolormesh(phi_f0_mesh, E_min_mesh, Z, cmap=cmap,
                           norm=mpl.colors.LogNorm())
    else:
        CS = ax.pcolormesh(phi_f0_mesh, E_min_mesh, Z, cmap=cmap)
    ax.set_xlabel("$\\phi_{f,0}$")
    ax.set_ylabel("$E_{\\mathrm{min}}$")
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(latex_quants[k])
    # Optional boundary curves
    if output_key in ["alpha_min", "phi_r", "t_crit"] and not fluid_flux:
        E_min_alpha_min_1 = alpha_min_1(phi_f0_arr, nu, Delta_p)
        ax.plot(phi_f0_arr, E_min_alpha_min_1, "--k", label="$\\alpha_{\\mathrm{min}} = 1$")
        ax.axvline(alpha_min_E_min(nu, Delta_p), linestyle=":", color="black",
                   label="$\\alpha_{\\mathrm{min}} = E_{\\mathrm{min}}$")
        ax.legend()
        ax.set_ylim(0.045, 1.005)
    fig.savefig(f"{plot_path}/_{subscripts[k]}.png", bbox_inches="tight")
    Z_df = pd.DataFrame(Z)
    fig.clf()
    Z_df.to_csv(f"{data_path}/ss_params_{subscripts[k]}.csv")
