"""
In this script, we compare different simulation results to see how similar they are.
"""

import os
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters and defining our paths
"""


def get_input_info(trial: str, sub_trial: str):
    """Retrieves information about the input directories and parameters.

    :param trial: The name of the trial.
    :param sub_trial: The name of the sub-trial.
    :return: A tuple containing the data path, plot path and parameter dict.
    """
    dir_path = f"resources/{trial}/{sub_trial}"
    data_path = f"{dir_path}/data"
    plot_path = f"{dir_path}/plots"
    param_file = open(f"{dir_path}/params.json")
    params = json.load(param_file)
    param_file.close()
    return data_path, plot_path, params


trial_1, sub_trial_1 = "long_steady_state", "v_0_1"
trial_2, sub_trial_2 = "linear", "const_Q"
data_path_1, plot_path_1, params_1 = get_input_info("long_steady_state", "v_0_1")
data_path_2, plot_path_2, params_2 = get_input_info("linear", "const_Q")

# The parameters from the simulation
N_x = params_1["comp"]["N_x"]
phi_f0 = params_1["ics"]["phi_f"]

# Retrieving the dataframes from each of the simulations
short_quants = ["phi"]
data_dict_1, data_dict_2 = {}, {}
for q in short_quants:
    q_array_1 = pd.read_csv(f"{data_path_1}/_{q}_xi.csv").to_numpy()[:, 2:]
    q_array_2 = pd.read_csv(f"{data_path_2}/{q}.csv").to_numpy()[:, 2:]
    data_dict_1[q] = q_array_1
    data_dict_2[q] = q_array_2


def rmse(arr_1: np.array, arr_2: np.array, conversion: tp.Callable = None):
    """Calculates the time-varying root mean squared error between the two arrays. If one
    array has more timesteps than the other but these numbers are divisible, then we
    record the timepoints at regular intervals. If there is a known conversion formula
    between the two arrays (e.g. a scaling for nondimensional purposes), we use the
    conversion to convert between the two (from the arr_2 coordinate system to the
    arr_1 coordinate system).

    :param arr_1: Array for first dataset (shape (N_x + 1, N_time + 1)).
    :param arr_2: Array for second dataset (shape (N_x + 1, N_time + 1)).
    :param conversion: A known conversion formula between the two arrays.
    :return: An array of length (N_time + 1) of mean squared errors.
    """
    if conversion is not None:
        arr_2 = conversion(arr_2)
    # Find the shapes of the arrays
    N_x_1, N_t_1 = arr_1.shape[0] - 1, arr_1.shape[1] - 1
    N_x_2, N_t_2 = arr_2.shape[0] - 1, arr_2.shape[1] - 1
    N_t = (N_t_1, N_t_2)
    arrs = (arr_1, arr_2)
    if N_x_1 != N_x_2:
        print("Exiting, arrays must have the same number of x points")
        return
    # Find which array has more timepoints and which has less (they may have the same)
    i_small, i_big = (0, 1) if N_t_2 > N_t_1 else (1, 0)
    N_t_small, N_t_big = N_t[i_small], N_t[i_big]
    if N_t_big % N_t_small != 0:
        print("Cannot take timepoints at regular intervals, exiting...")
        return
    # Calculate the factor and take timepoints of arr_big at regular intervals
    factor = int(N_t_big / N_t_small)
    arr_small, arr_big = arrs[i_small], arrs[i_big]
    arr_big_reduced = arr_big[:, ::factor]
    # Find the root mean squared error
    return np.sqrt(np.mean((arr_big_reduced - arr_small) ** 2, axis=0))


epsilon = 0.01
conversion_formula = lambda phi_f1: phi_f0 + epsilon * phi_f1
rmse_array = rmse(data_dict_1["phi"], data_dict_2["phi"], conversion_formula)
print(rmse_array)

fig, ax = plt.subplots(1, 1, figsize=(8, 10/3))
delta_t = max(params_1["comp"]["delta_t"], params_2["comp"]["delta_t"])
times = np.linspace(0, (len(rmse_array) - 1) * delta_t, len(rmse_array))
ax.plot(times, rmse_array, color="black", lw=2.5)
ax.set_xlabel("Time")
ax.set_ylabel("Root mean squared error")
comparisons_path = f"comparisons/{trial_1}_vs_{trial_2}"
if not os.path.exists(comparisons_path):
    os.makedirs(comparisons_path)
fig.savefig(f"{comparisons_path}/rmse.png", bbox_inches="tight")
