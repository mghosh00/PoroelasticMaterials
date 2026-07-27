"""
In this file, we compute the quasi-steady state for a given set of parameter values and save
the results to .csv files.
"""

import os
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas.errors
import scipy.integrate as si
import pandas as pd
import json

from steady_state import SteadyState

warnings.simplefilter(action='ignore', category=pandas.errors.PerformanceWarning)
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

plot_coord = 'X'

"""
Reading in our parameters and defining our paths
"""
parent = "nonphys"
trial = "qss"
sub_trial = "phi_f0_0_4_gamma_0_25"
dir_path = f"resources/{parent}/{trial}/{sub_trial}"
data_path = f"{dir_path}/data"
param_file = open(f"{dir_path}/params.json")
params = json.load(param_file)


def convert_to_X(_phi_f: np.array, _xi_arr: np.array, _a: float, _phi_f0: float):
    """Converts the array _phi_f into (X, t) coordinates from (x, t) coordinates
    using interpolation and the displacement array u_s.

    :param _phi_f: The porosity array in (x, t) coordinates.
    :param _xi_arr: The xi array.
    :param _a: The left boundary.
    :param _phi_f0: The initial porosity.
    :return: The porosity array in (X, t) coordinates.
    """
    _x_arr = _a + (1 - _a) * _xi_arr
    integral = np.array([0] + [si.simpson(_phi_f[:j], _x_arr[:j]) for j in range(1, len(_x_arr))])
    _u_s = _a + (integral - _phi_f0 * (_x_arr - _a)) / (1 - _phi_f0)
    X = _x_arr - _u_s
    _phi_f_X = np.interp(X, _xi_arr, _phi_f)
    return _phi_f_X


def phi_f_qss(_xi_arr: np.array, _t: float,
              _E_min: float, _params: dict, _plot_coord: str):
    """Calculates the quasi-steady expression for phi_f for the case in which the
    weakening timescale is much longer than the other timescales, and we are in this
    regime. Provided c has converged to its steady state of a constant profile, the
    leading-order contribution to the Young's modulus takes the form
    E_0(t) = E_min + (1 - E_min) * e^{-t}.

    :param _xi_arr: The spatial coordinate array.
    :param _t: The timestep.
    :param _E_min: The minimal Young's modulus value.
    :param _params: All other parameters from the params dict.
    :param _plot_coord: The coordinate which we plot against.
    :return: The quasi-steady expressions for the porosity and phase-averaged velocity.
    """
    E_0 = _E_min + (1 - _E_min) * np.exp(-_t)
    # The next line is to ensure that the time-varying function goes into the
    # steady state calculation
    _params["phys"]["E_min"] = E_0
    quasi_steady_state = SteadyState(_params, _xi_arr)
    _phi_f_qss, _a_qss, _B_qss = quasi_steady_state.solve_analytic()
    _v_qss = quasi_steady_state.Q_f
    # _phi_f_qss is in xi coordinates, but we want to convert it to x coordinates
    if _plot_coord == "x":
        _x_arr = _a_qss + (1 - _a_qss) * _xi_arr
        _coord_arr = _x_arr
    elif _plot_coord == "X":
        _coord_arr = _xi_arr
        _phi_f0 = _params["ics"]["phi_f"]
        _phi_f_qss = convert_to_X(_phi_f_qss, _xi_arr, _a_qss, _phi_f0)
    else:
        _coord_arr = _xi_arr
    return _coord_arr, _phi_f_qss, _a_qss, _v_qss


N_x, N_time, delta_t = params["comp"]["N_x"], params["comp"]["N_time"], params["comp"]["delta_t"]
E_min = params["phys"]["E_min"]
X_arr = np.linspace(0, 1, params["comp"]["N_x"] + 1)
phi_f_df = pd.DataFrame({'X': X_arr})
old_phi_f = params["ics"]["phi_f"] * np.ones(N_x + 1)
for n in range(N_time):
    t = n * delta_t
    print("Time:", np.round(t, 3))
    _, phi_f, _, _ = phi_f_qss(X_arr, t, E_min, params, plot_coord)
    if old_phi_f[-1] < phi_f[-1]:
        print(f"Pores have closed at t = {np.round(t, 3)}")
        break
    phi_f_df[t] = phi_f
    old_phi_f = phi_f

if not os.path.exists(data_path):
    os.makedirs(data_path)

phi_f_df.to_csv(f"{data_path}/_phi_f_qss_X.csv")
