"""
In this file, we compare the analytic steady state to the numerical steady state
from one of our simulations. This steady state assumes that the Young's modulus
has reached a uniform steady state of E_min (so this is long-term analysis). We produce
comparative plots (and calculate a sum of squares error).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.integrate as si
import scipy.optimize as so
import json

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters and defining our paths
"""
trial = "long_steady_state"
sub_trial = "v_0_1_steprev"
dir_path = f"resources/{trial}/{sub_trial}"
data_path = f"{dir_path}/data"
plot_path = f"{dir_path}/plots"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)

# The parameters from the simulation
N_x = params["comp"]["N_x"]

phi_f0 = params["ics"]["phi_f"]

L = params["phys"]["L"]
nu = params["phys"]["nu"]
mu = params["phys"]["mu"]
E_min = params["phys"]["E_min"]
D_m = params["phys"]["D_m"]

k_0 = params["scales"]["k"]
E_star = params["scales"]["E"]
v_star = params["scales"]["v"]

v_final = params["v"]["v_final"]

c_left = params["bcs"]["c_left"]

param_file.close()

t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
t_c = L ** 2 / D_m


def final_array(path: str):
    """Gets the final profile for the quantity from the given filepath.

    :param path: The path to the .csv file containing time series data for the quantity.
    :return: The np.array of the quantity at the last timestep.
    """
    f_array = pd.read_csv(path).to_numpy()
    f_final = f_array[1:, -1]
    return f_final[~np.isnan(f_final)]


# Find the final value of a(t) --- this comes from the u_s_array
u_s_final = final_array(f"{data_path}/u_s.csv")
a_final = u_s_final[0]

phi_f_final = final_array(f"{data_path}/phi.csv")
c_final = final_array(f"{data_path}/c.csv")
x = np.linspace(1 - (len(phi_f_final) - 1) / N_x, 1, len(phi_f_final))

# From steady state analysis, once E -> E_min, we can determine phi_f, a and
# c. We must solve for phi_f and a simultaneously to determine an integration
# constant, B. c can then be recovered.


def F(_phi_f: np.array, *args):
    """The relation for the analytic
    solution of the steady state of phi_f.

    :param _phi_f: The porosity array.
    :param args: The remaining arguments.
    :return: The right hand side of the steady state relation.
    """
    _phi_f0 = args[0]
    _nu = args[1]
    _B = args[2]
    _factor = args[3]
    _x = args[4]
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    factor1 = (1 - _phi_f0) ** 2
    term1 = 1 / (3 * (1 - _phi_f) ** 3) - 3 / (2 * (1 - _phi_f) ** 2) + 3 / (1 - _phi_f) + np.log(1 - _phi_f)
    factor2 = 1 - 2 * _nu
    term2 = 1 / (1 - _phi_f) + 3 * np.log(1 - _phi_f) - 3 * (1 - _phi_f) + 1 / 2 * (1 - _phi_f) ** 2
    return (factor1 * term1 + factor2 * term2) / denominator + _factor * _x - _B


def calculate_phi(_x: np.array, _phi_f0: float, _nu: float, _t_phi: float, _t_v: float,
                  _v_inf: float, _E_min: float):
    """Uses scipy.optimize to invert the following relation between phi and x:
    B - t_{phi} / t_{v} * v_{\\infty} / E_{min} * \\phi_{f,0}^3 / (1 - \\phi_{f,0}) * x = F(\\phi_{f})

    :param _x: The spatial coordinate.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _t_phi: The poroelastic timescale.
    :param _t_v: The advective timescale.
    :param _v_inf: The final value of the velocity.
    :param _E_min: The minimal Young's modulus.
    :return: The array phi_f.
    """
    factor = (_t_phi * _v_inf * _phi_f0 ** 3) / (_t_v * _E_min * (1 - _phi_f0))
    B = factor + F(0, _phi_f0, _nu, 0, factor, 1)
    phi_f_initial = np.array([phi_f0] * len(_x))
    phi_f = so.fsolve(F, phi_f_initial,
                      args=(_phi_f0, _nu, B, factor, _x))
    return phi_f


def calculate_a(_phi_f: np.array, _phi_f0: float, _x: np.array):
    """Calculates the value of the left boundary given the porosity.

    :param _phi_f: The porosity array.
    :param _phi_f0: The initial porosity.
    :param _x: The spatial coordinate array.
    :return: The integral of a function of porosity.
    """
    dx = _x[1] - _x[0]
    integrand = _phi_f
    integral = si.simpson(integrand, _x, dx=dx)
    return _phi_f0 - integral


def calculate_c(_phi_f: np.array, _x: np.array,
                _t_c: float, _t_v: float, _v_inf: float, _c_left: float):
    """Calculates the steady state value for c given the porosity.

    :param _phi_f: The porosity array.
    :param _x: The spatial coordinate array.
    :param _t_c: The solute timescale.
    :param _t_v: The phase-averaged velocity timescale.
    :param _v_inf: The final value of the velocity.
    :param _c_left: The left value of the solute concentration.
    :return: The predicted steady state profile for the solute concentration.
    """
    dx = _x[1] - _x[0]
    exponent_list = [0]
    for i in range(len(_x) - 1):
        integral_i = si.simpson(1 / _phi_f[:i + 1], _x[:i + 1], dx=dx)
        exponent_list.append(integral_i)
    exponent = np.array(exponent_list) * _t_c / _t_v * _v_inf
    return _c_left * np.exp(exponent)


phi_f_ss = calculate_phi(x, phi_f0, nu, t_phi, t_v, v_final, E_min)
a_ss = calculate_a(phi_f_ss, phi_f0, x)
c_ss = calculate_c(phi_f_ss, x, t_c, t_v, v_final, c_left)

# Finally, we plot the prediction and the actual values of the final state for the
# Young's modulus. We'll also plot the linearised version of phi alongside the actual
# phi profile.

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 20/3))
ax_phi, ax_E = axs

# Plotting for phi
ax_phi.plot(x, phi_f_final, "-k", label="True")
ax_phi.plot(x, phi_f_ss, "--r", label="Analytic steady state")
ax_phi.legend()
ax_phi.set_xlabel("$x$")
ax_phi.set_ylabel("$\\phi_f$")

# Plotting for c
ax_E.plot(x, c_final, "-k", label="True")
ax_E.plot(x, c_ss, "--r", label="Analytic steady state")
ax_E.legend()
ax_E.set_xlabel("$x$")
ax_E.set_ylabel("$c$")

print(x)

# Saving the figure
fig.savefig(f"{plot_path}/long_steady_state.png", bbox_inches="tight")

print(f"True value of a_inf: {a_final}")
print(f"Predicted value of a_inf: {a_ss}")
