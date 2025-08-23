import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.integrate as si
import json

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
plot_path = f"{dir_path}/plots"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)

# The parameters from the simulation
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

# First, we take the final state of phi and transform it onto a line (linear
# approximation of phi)

phi_final = final_array(f"{data_path}/phi.csv")
E_final = final_array(f"{data_path}/E.csv")
x = np.linspace(1 - 0.01 * len(phi_final), 1, len(phi_final))

# Alpha is the gradient of the line of best fit, beta is the intercept
beta, alpha = np.polyfit(x, phi_final, 1)
phi_lin = alpha + beta * x

# From steady state analysis, if c -> 0, and we are at a steady state, then we have
# an explicit relation between the Young's modulus and the porosity. Performing the
# linear interpolation allows for a formula for E in terms of explicit functions.


def g(_phi_f: np.array, _phi_f0: float, _nu: float):
    """The effective stress function

    :param _phi_f: The porosity array.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :return: The effective stress.
    """
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    term1 = (1 - _phi_f0) ** 2 / (1 - _phi_f)
    term2 = - 2 * _nu * (1 - _phi_f0)
    term3 = - (1 - 2 * _nu) * (1 - _phi_f)
    return (term1 + term2 + term3) / denominator


def integral_perm_linear(_phi_f: np.array, _phi_f0: float, _beta: float):
    """The explicit form for the integral of a variation of the effective permeability.

    :param _phi_f: The linear porosity array.
    :param _phi_f0: The initial porosity.
    :return: The integral of a function of the porosity.
    """
    phi_f_right = _phi_f[-1]
    term1 = - 1 / (2 * phi_f_right ** 2)
    term2 = 2 / phi_f_right
    term3 = np.log(phi_f_right)
    term4 = 1 / (2 * _phi_f ** 2)
    term5 = - 2 / _phi_f
    term6 = - np.log(_phi_f)
    factor = _beta * _phi_f0 ** 3 / (1 - _phi_f0)
    return factor * (term1 + term2 + term3 + term4 + term5 + term6)


def integral_perm_nonlinear(_phi_f: np.array, _phi_f0: float, _x: np.array):
    """Calculates the numerical integral of a variation of the effective permeability.

    :param _phi_f: The porosity array.
    :param _phi_f0: The initial porosity.
    :param _x: The spatial coordinate array.
    :return: The integral of a function of porosity.
    """
    factor = _phi_f0 ** 3 / (1 - _phi_f0)
    dx = _x[1] - _x[0]
    integrand = (1 - _phi_f) ** 2 / _phi_f ** 3
    integral = si.simpson(integrand, _x, dx=dx)
    return factor * integral


def calculate_E(_phi_f: np.array, _phi_f0: float, _x: np.array, _nu: float,
                _t_phi: float, _t_v: float, _v: float, E_1: float,
                linear: bool = False, _beta: float = 0.0):
    """Calculates the steady state value for E. Note that we must pass the true value
    for E at one point (chosen to be x = 1 here for convenience).

    :param _phi_f: The porosity array.
    :param _phi_f0: The initial porosity.
    :param _x: The spatial coordinate array.
    :param _nu: The Poisson's ratio.
    :param _t_phi: The porosity timescale.
    :param _t_v: The phase-averaged velocity timescale.
    :param _v: The phase-averaged velocity.
    :param E_1: The true final value of the Young's modulus at x = 1.
    :param linear: Whether we are working with a linear porosity or not.
    :param _beta: The gradient of _phi_f if it is linear.
    :return: The predicted steady state profile for the Young's modulus.
    """
    _g = g(_phi_f, _phi_f0, _nu)
    A = E_1 * _g[-1]
    if linear:
        _int = integral_perm_linear(_phi_f, _phi_f0, _beta)
    else:
        _int = integral_perm_nonlinear(_phi_f, _phi_f0, _x)
    return 1 / _g * (A + _t_phi * _v * _int / _t_v)


E_steady_state_lin = calculate_E(phi_lin, phi_f0, x, nu, t_phi,
                                 t_v, v_final, float(E_final[-1]),
                                 linear=True, _beta=beta)
E_steady_state_nonlin = calculate_E(phi_final, phi_f0, x, nu, t_phi,
                                    t_v, v_final, float(E_final[-1]))

# Finally, we plot the prediction and the actual values of the final state for the
# Young's modulus. We'll also plot the linearised version of phi alongside the actual
# phi profile.

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 20/3))
ax_phi, ax_E = axs

# Plotting for phi
ax_phi.plot(x, phi_final, "-k", label="True")
ax_phi.plot(x, phi_lin, "--r", label="Linearised")
ax_phi.legend()
ax_phi.set_xlabel("$x$")
ax_phi.set_ylabel("$\\phi_f$")

# Plotting for E
ax_E.plot(x, E_final, "-k", label="True")
ax_E.plot(x, E_steady_state_lin, "--r", label="Predicted: linear $\\phi_f$")
ax_E.plot(x, E_steady_state_nonlin, "--", color='cyan', label="Predicted: nonlinear $\\phi_f$")
ax_E.legend()
ax_E.set_xlabel("$x$")
ax_E.set_ylabel("$E$")

# Saving the figure
fig.savefig(f"{plot_path}/steady_state.png", bbox_inches="tight")
