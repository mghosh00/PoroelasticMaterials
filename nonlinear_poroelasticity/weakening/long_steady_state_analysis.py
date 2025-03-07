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
sub_trial = "v_0_1_sigma_0_1"
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
sigma_l = params["bcs"]["sigma_left"]

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
    f_final = f_array[0:, -1]
    return f_final[~np.isnan(f_final)]


# Find the final value of a(t) --- this comes from the u_s_array
u_s_final = final_array(f"{data_path}/u_s.csv")
a_final = u_s_final[0]

phi_f_final = final_array(f"{data_path}/phi.csv")
c_final = final_array(f"{data_path}/c.csv")
x = np.linspace(1 - (len(phi_f_final) - 1) / N_x, 1, len(phi_f_final))
xi = np.linspace(0, 1, N_x + 1)

# From steady state analysis, once E -> E_min, we can determine phi_f, a and
# c. We must solve for phi_f and a simultaneously to determine an integration
# constant, B. c can then be recovered.


def get_phi_l(_sigma_l, _E_min, _phi_f0, _nu):
    """Finds the value of phi_f on the left.

    :param _sigma_l: The value of sigma' on the left.
    :param _E_min: The steady state value of the Young's modulus.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :return: The value of the porosity on the left.
    """
    b = (1 + _nu) * (1 - 2 * _nu) * _sigma_l / _E_min + 2 * _nu
    discriminant = b ** 2 + 4 * (1 - 2 * _nu)
    factor = (1 - _phi_f0) / (2 * (1 - 2 * _nu))
    return 1 - factor * (discriminant ** (1 / 2) - b)


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
    _a = args[4]
    _xi = args[5]
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    factor1 = (1 - _phi_f0) ** 2
    term1 = 1 / (3 * (1 - _phi_f) ** 3) - 3 / (2 * (1 - _phi_f) ** 2) + 3 / (1 - _phi_f) + np.log(1 - _phi_f)
    factor2 = 1 - 2 * _nu
    term2 = 1 / (1 - _phi_f) + 3 * np.log(1 - _phi_f) - 3 * (1 - _phi_f) + 1 / 2 * (1 - _phi_f) ** 2
    _x = _a + (1 - _a) * _xi
    return (factor1 * term1 + factor2 * term2) / denominator + _factor * _x - _B


def guess_B(_phi_f0: float, _nu: float, _factor: float, _phi_l: float,
            _a: float):
    """Given the above parameters, finds the value that B must take.

    :param _phi_f0: Initial porosity.
    :param _nu: Poisson's ratio.
    :param _factor: A combination of parameters.
    :param _phi_l: Value of phi on the left (final).
    :param _a: Position of the left boundary.
    :return: The guess for B.
    """
    return F(_phi_l, _phi_f0, _nu, 0, _factor, _a, 0)


def guess_phi(_xi: np.array, _phi_f0: float, _nu: float, _factor: float,
              _a: float, _B: float):
    """Uses scipy.optimize to invert the relation between phi and xi.

    :param _xi: The spatial coordinate.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _factor: A combination of parameters.
    :param _a: The guess for the left boundary.
    :param _B: The guess for the integration constant, B.
    :return: The array phi_f.
    """
    phi_f_initial = np.array([phi_f0] * len(_xi))
    phi_f_list = []
    for i in range(len(_xi)):
        phi_f_i = so.fsolve(F, phi_f_initial[i],
                            args=(_phi_f0, _nu, _B, factor, _a, _xi[i]))
        phi_f_list.append(phi_f_i[0])
    phi_f = np.array(phi_f_list)
    return phi_f


def guess_a(_phi_f: np.array, _phi_f0: float, _xi: np.array):
    """Calculates the value of the left boundary given the porosity.

    :param _phi_f: The porosity array.
    :param _phi_f0: The initial porosity.
    :param _xi: The spatial coordinate array.
    :return: The integral of a function of porosity.
    """
    dxi = _xi[1] - _xi[0]
    integrand = _phi_f
    integral = si.simpson(integrand, _xi, dx=dxi)
    return (_phi_f0 - integral) / (1 - integral)


def calculate_c(_phi_f: np.array, _a: float, _xi: np.array,
                _t_c: float, _t_v: float, _v_inf: float, _c_left: float):
    """Calculates the steady state value for c given the porosity.

    :param _phi_f: The porosity array.
    :param _a: The left boundary.
    :param _xi: The spatial coordinate array.
    :param _t_c: The solute timescale.
    :param _t_v: The phase-averaged velocity timescale.
    :param _v_inf: The final value of the velocity.
    :param _c_left: The left value of the solute concentration.
    :return: The predicted steady state profile for the solute concentration.
    """
    dxi = _xi[1] - _xi[0]
    exponent_list = []
    for i in range(len(_xi)):
        integral_i = si.simpson((1 - _a) / _phi_f[:i + 1], _xi[:i + 1], dx=dxi)
        exponent_list.append(integral_i)
    exponent = np.array(exponent_list) * _t_c / _t_v * _v_inf
    return _c_left * np.exp(exponent)


def calculate_u_s(_phi_f: np.array, _a: float, _xi: np.array, _phi_f0: float):
    """Calculates the steady state for the displacement given the porosity
    and left boundary.

    :param _phi_f: Steady state porosity profile.
    :param _a: Left boundary.
    :param _xi: Spatial coordinate.
    :param _phi_f0: Initial porosity.
    :return: The steady state displacement.
    """
    multiplier = (1 - _a) / (1 - _phi_f0)
    dxi = _xi[1] - _xi[0]
    integral_list = []
    for i in range(len(_xi)):
        integral_i = si.simpson(_phi_f[:i + 1], _xi[:i + 1], dx=dxi)
        integral_list.append(integral_i)
    integral_arr = np.array(integral_list)
    return multiplier * (integral_arr - _phi_f0 * _xi) + _a


phi_l = get_phi_l(sigma_l, E_min, phi_f0, nu)
factor = (t_phi * v_final * phi_f0 ** 3) / (t_v * E_min * (1 - phi_f0))
# We need a loop to converge on solutions for phi_f, a and B
phi_f_guess = np.array([phi_f0] * len(xi))
# Pre-calculated error requirement (and going one order of magnitude lower)
tol = 1e-7 * N_x
i = 0
while True:
    print(f"Iteration {i}")
    a_guess = guess_a(phi_f_guess, phi_f0, xi)
    print(f"a_{i}: {a_guess}")
    B_guess = guess_B(phi_f0, nu, factor, phi_l, a_guess)
    print(f"B_{i}: {B_guess}")
    phi_f_guess_new = guess_phi(xi, phi_f0, nu, factor, a_guess, B_guess)
    sum_squares = ((phi_f_guess_new - phi_f_guess) ** 2).sum()
    print(f"sum_sq_{i}: {sum_squares}")
    phi_f_guess = phi_f_guess_new
    print(f"phi_f_{i}: {phi_f_guess}")
    if sum_squares < tol:
        phi_f_ss = phi_f_guess
        a_ss = a_guess
        break
    i += 1


c_ss = calculate_c(phi_f_ss, a_ss, xi, t_c, t_v, v_final, c_left)
u_s_ss = calculate_u_s(phi_f_ss, a_ss, xi, phi_f0)

# Finally, we plot the prediction and the actual values of the final state for the
# Young's modulus. We'll also plot the linearised version of phi alongside the actual
# phi profile.

nrows, ncols = 3, 1
fig, axs = plt.subplots(nrows, ncols, sharex=True, figsize=(8, nrows * 10/3))
ax_phi, ax_c, ax_us = axs

# Plotting for phi
ax_phi.plot(xi, phi_f_final, "-k", label="True")
ax_phi.plot(xi, phi_f_ss, "--r", label="Analytic steady state")
ax_phi.legend()
ax_phi.set_xlabel("$\\xi$")
ax_phi.set_ylabel("$\\phi_f$")

# Plotting for c
ax_c.plot(xi, c_final, "-k", label="True")
ax_c.plot(xi, c_ss, "--r", label="Analytic steady state")
ax_c.legend()
ax_c.set_xlabel("$\\xi$")
ax_c.set_ylabel("$c$")

# Plotting for u_s
ax_us.plot(xi, u_s_final, "-k", label="True")
ax_us.plot(xi, u_s_ss, "--r", label="Analytic steady state")
ax_us.legend()
ax_us.set_xlabel("$\\xi$")
ax_us.set_ylabel("$u_s$")

# Saving the figure
fig.savefig(f"{plot_path}/long_steady_state.png", bbox_inches="tight")

print(f"True value of a_inf: {a_final}")
print(f"Predicted value of a_inf: {a_ss}")
