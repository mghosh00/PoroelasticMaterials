"""
In this script, we animate the porosity solution from FEniCS (from nondim_weakening.py) and
compare it to two separate boundary layer predictions near x = 1, the first for early
times and the second for later times. The prediction for early times looks like:

\\phi_{f} = \\phi_{f,0} - v(1 - \\phi_{f,0})(2\\sqrt{t}{\\pi D_{1}}\\exp{(-\\frac{(1-x)^{2}}{4D_{1}t})}
                            - \\frac{1-x}{D_{1}}(1 - \\erf{(\\frac{1-x}{2\\sqrt{D_{1}t}})})),

where \\phi_{f,0} is the initial porosity, v is the nondimensional flux and D_{1} is a diffusive
coefficient for the BL equation.

For late times, the boundary layer in x produces the following time-independent prediction:

\\phi_{f} = (\\frac{v(1-x)}{D_{2}})^{\\frac{1}{4}}.

We will compare the result from the numerics to these two boundary layer predictions.
"""


import os
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import pandas as pd
import json
from PIL import Image
import time

from fenics import Expression

from nonlinear_poroelasticity.weakening.scripts import Quantity, SteadyState

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
parent = "phys"
trial = "porous_polymer"
sub_trial = "Delta_p_0_1"
path = f"resources/{parent}/{trial}/{sub_trial}"
param_file = open(f"{path}/params.json")
params = json.load(param_file)

# Whether we produce a .gif or set of traces
gif = False
# Whether we create panels
panels = True
# Whether we plot on log-log axes or not
log = False
log_text = "_log" if log else ""

# Whether we'll plot on a fixed domain or not
fixed_domain = False
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"
plot_coord_tex = f"log$(1 - ${plot_coord_tex})" if log else plot_coord_tex

# Whether we read the analytic solutions for a and v from a dataframe or not
read_a_v = False

"""
Computational parameters
"""

# Size of time step
delta_tau = params["comp"]["delta_tau"]

# Number of time steps
N_time = params["comp"]["N_time"]

# Number of mesh points
N_x = params["comp"]["N_x"]

"""
Define model parameters
"""

# Length of domain, L
L = params["phys"]["L"]

# Initial porosity, \\phi_{f,0}
phi_f0 = params["ics"]["phi_f"]

# Poisson ratio, viscosity and weakening
nu = params["phys"]["nu"]
mu = params["phys"]["mu"]
beta_E = params["phys"]["beta_E"]
E_min = params["phys"]["E_min"]

# Permeability scale
k_0 = params["scales"]["k"]

# Solute concentration, Young's modulus and velocity scales
c_star = params["scales"]["c"]
E_star = params["scales"]["E"]
v_star = params["scales"]["v"]
if "Q_f" in params:
    Q_f = params["Q_f"]["Q_f_final"]
    fluid_flux = True
    t_v = L / v_star
else:
    Delta_p = params["bcs"]["Delta p"]
    fluid_flux = False
    t_v = (mu * L ** 2) / (k_0 * params["bcs"]["Delta p"] * E_star)
    params["scales"]["v"] = L / t_v

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_E = 1 / (beta_E * c_star)
t_sc = params["scales"]["t"]

epsilon_E = t_phi / t_E

# The collective "diffusion coefficient" for the early equation
D_phi_early = (1 * (1 - nu) * t_v) / (t_phi * (1 + nu) * (1 - 2 * nu))

# The diffusion coefficient for the system with applied pressure drop
D_phi_early_pd = (1 - nu) / (1 + nu) / (1 - 2 * nu)

# The collective "diffusion coefficient" for the late equation
D_phi_late = ((1 * (1 - phi_f0) * ((1 - phi_f0) ** 2 + 1 - 2 * nu) * t_v) /
              (8 * t_phi * (1 + nu) * (1 - 2 * nu) * phi_f0 ** 3))

"""
Read in quantity .csv files.
"""
short_quants = ["phi"]
latex_quants = ["log$(\\phi_{f})$" if log else "$\\phi_{f}$"]
colours = ["blue"]
data_path = f"{path}/data"
plot_path = f"{path}/plots"
file_names = [f"{data_path}/_{q}_{plot_coord}.csv" for q in short_quants]
data_dict = {}
for i, name in enumerate(short_quants):
    data_arr = pd.read_csv(file_names[i], header=None).to_numpy()
    times = np.array(data_arr[0, 2:], dtype=float)
    times[0] = 0.0
    coord_arr = np.array(data_arr[1:, 1], dtype=float)
    quant_arr_nan = np.array(data_arr[1:, 2:], dtype=float)
    data_dict[name] = quant_arr_nan

# t_tau = params["comp"]["t(tau)"] if "t(tau)" in params["comp"] else "tau"
# t_expr = Expression(t_tau, degree=1, tau=0.0, delta_tau=delta_tau, N_time=N_time)
# times = [0.0]
# for n in range(N_time):
#     times.append(float(t_expr(0.0)))
#     t_expr.tau += delta_tau
# times = np.array(times)
coord_arr[0] = 0
coord_arr[-1] = 1

# if log:
#     coord_arr = 1 - coord_arr


"""
Defining the various analytic solutions to parts of the problem
"""


def phi_f_bl_early(_x: np.array, _t: float, _phi_f0: float, _Q_f: float,
                   _D_phi_early: float):
    """Returns the np.array solution in the early boundary layer. This is
    described at the top of the file.

    :param _x: The spatial coordinate array.
    :param _t: The specific timepoint.
    :param _phi_f0: The initial porosity.
    :param _Q_f: The nondimensional volume flux.
    :param _D_phi_early: The diffusion coefficient for the problem.
    :return: The porosity array at this timepoint within the early BL.
    """
    term2 = 2 * np.sqrt(_t / (np.pi * _D_phi_early)) * np.exp(- (1 - _x) ** 2 / (4 * _D_phi_early * _t))
    term3 = - (1 - _x) / _D_phi_early * (1 - ss.erf((1 - _x) / (2 * np.sqrt(_D_phi_early * _t))))
    return _phi_f0 - _Q_f * (1 - _phi_f0) * (term2 + term3)


def phi_f_bl_early_pressure_drop(_x: np.array, _t: float, _phi_f0: float, _nu: float,
                                 _Delta_p: float, _D_phi_early: float, _epsilon_E: float):
    """Returns the porosity in the early boundary layer when there is an applied
    pressure drop.

    :param _x: The spatial array.
    :param _t: The timepoint.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _Delta_p: The pressure drop.
    :param _D_phi_early: The diffusion coefficient.
    :param _epsilon_E: The ratio of the poroelastic and weakening timescales.
    :return: The porosity in this boundary layer.
    """
    multiplier = (1 - _phi_f0) * (1 + _nu) * (1 - 2 * _nu) * _Delta_p / (1 - _nu)
    inner = 1 - ss.erf((1 - _x) / (2 * np.sqrt(_D_phi_early * _t / _epsilon_E)))
    return _phi_f0 - multiplier * inner


def phi_f_bl_late(_x: np.array, _Q_f: float, _D_phi_late: float):
    """Returns the np.array solution in the late boundary layer. This is
    described at the top of the file.

    :param _x: The spatial coordinate array.
    :param _Q_f: The nondimensional volume flux.
    :param _D_phi_late: The diffusion coefficient for the problem.
    :return: The porosity array at this timepoint within the late BL.
    """
    return (_Q_f * (1 - _x) / _D_phi_late) ** (1 / 4)


def phi_f_bl_late_longer(_x: np.array, _v: float, _D_phi_late: float,
                         _phi_f0: float, _nu: float):
    """Returns the np.array solution in the late boundary layer. This is
    described at the top of the file, but with an extra term.

    :param _x: The spatial coordinate array.
    :param _v: The nondimensional volume flux.
    :param _D_phi_late: The diffusion coefficient for the problem.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :return: The porosity array at this timepoint within the late BL.
    """
    term1 = (_v * (1 - _x) / _D_phi_late) ** (1 / 4)
    factor2 = - 2 * (2 * (1 - _phi_f0) ** 2 + 1 - 2 * _nu) / (5 * ((1 - _phi_f0) ** 2 + 1 - 2 * _nu))
    term2 = (_v * (1 - _x) / _D_phi_late) ** (1 / 2)
    return term1 + factor2 * term2


def phi_f_lin_elastic(_x_arr: np.array, _t: float, _Q_f: float,
                      _phi_f0: float, _D_phi: float, n_terms: int = 1000):
    """Calculates the analytic Fourier series expression for the porosity when
    we have zero solid stress on the RIGHT boundary. Q_f must also be constant
    in time.

    :param _x_arr: The spatial coordinate array.
    :param _t: The current timepoint.
    :param _Q_f: The imposed fluid flux.
    :param _phi_f0: The initial porosity.
    :param _D_phi: The diffusion coefficient for the porosity equation.
    :param n_terms: The number of terms in the Fourier series expansion.
    :return: The porosity at the current timepoint.
    """
    _phi_f1 = - (1 - _phi_f0) / _D_phi * _Q_f * _x_arr
    for m in range(n_terms):
        factor = 8 / ((2 * m + 1) * np.pi) ** 2 * (1 - _phi_f0) / _D_phi * _Q_f
        summand = np.cos((m + 1 / 2) * np.pi * (1 - _x_arr)) * np.exp(- ((m + 1 / 2) * np.pi) ** 2 * _D_phi * _t)
        _phi_f1 += factor * summand
    return _phi_f0 + _phi_f1


def phi_f_quasi_steady_late(_xi_arr: np.array, _t: float, _t_E: float,
                            _E_min: float, _params: dict, _fixed_domain: bool):
    """Calculates the quasi-steady expression for phi_f for the case in which the
    weakening timescale is much longer than the other timescales, and we are in this
    regime. Provided c has converged to its steady state of a constant profile, the
    leading-order contribution to the Young's modulus takes the form
    E_0(t) = E_min + (1 - E_min) * e^{-t * t_sc / t_{E}}.

    :param _xi_arr: The spatial coordinate array.
    :param _t: The timestep.
    :param _t_E: The weakening timescale.
    :param _E_min: The minimal Young's modulus value.
    :param _params: All other parameters from the params dict.
    :return: The quasi-steady expressions for the porosity and phase-averaged velocity.
    """
    _t_sc = _params["scales"]["t"]
    E_0 = _E_min + (1 - _E_min) * np.exp(-_t)
    # The next line is to ensure that the time-varying function goes into the
    # steady state calculation
    _params["phys"]["E_min"] = E_0
    quasi_steady_state = SteadyState(_params, _xi_arr)
    _phi_f_qss, _a_qss, _B_qss = quasi_steady_state.solve_analytic()
    _v_qss = quasi_steady_state.Q_f
    # _phi_f_qss is in xi coordinates, but we want to convert it to x coordinates
    if fixed_domain:
        _x_arr = _a_qss + (1 - _a_qss) * _xi_arr
        _coord_arr = _x_arr
    else:
        _coord_arr = _xi_arr
    return _coord_arr, _phi_f_qss, _a_qss, _v_qss


"""
Loop over time steps and plot each frame
"""

# We store each frame separately before making the .gif file
if not os.path.exists(f"{plot_path}/frames"):
    os.makedirs(f"{plot_path}/frames")


images = []
tau_period = 0.0013

# Set up the colorbars and label the plots
mins = np.array([np.nanmin(data_dict[name]) for name in short_quants])
maxes = np.array([np.nanmax(data_dict[name]) for name in short_quants])

ranges = maxes - mins
if log:
    xmin, xmax = np.log(np.min(coord_arr)), np.log(np.max(coord_arr))
    mins, maxes = np.log(mins), np.log(maxes)
    print(mins, maxes)
else:
    mins -= 0.1 * ranges
    maxes += 0.1 * ranges
    xmin, xmax = coord_arr[0], coord_arr[-1]
    xmin -= 0.1 * (coord_arr[-1] - coord_arr[0])
    xmax += 0.1 * (coord_arr[-1] - coord_arr[0])
nrows, ncols = 1, 1

if not gif:
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, (10 * nrows)/3), sharex=True)
    plt.subplots_adjust(wspace=1.0 * (ncols - 1))
    axs_list = [axs]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(times[-1]))
    phi_f = Quantity(latex_quants[0], "Blues", 0)
    phi_f.mesh_fixed = np.log(1 - coord_arr) if log else coord_arr
    phi_f.set_ax(axs_list[0])

early_label = "Early time"
lin_label = "Linear elasticity (analytic)"
late_label = "$\\tilde{\\Phi}_{f,0}$"
late_extra_label = "$\\tilde{\\Phi}_{f,0} + \\tilde{\\epsilon}\\tilde{\\Phi}_{f,1}$"
quasi_steady_label = "Quasi-steady"
fenics_label = "Numerical"
n_end = int(N_time * delta_tau / tau_period) + 1
# n_end = 50

if panels:
    nrows, ncols = 4, 3
    num_panels = nrows * ncols
    panels_n_list = list(np.linspace(0, n_end - 1, num=num_panels).astype(int))
    print(panels_n_list)
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 13), sharex="col", sharey="row")
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    axs_list = [axs[i][j] for i in range(nrows) for j in range(ncols)]

t_1, t_2 = 0.04, 1
for n in range(n_end):
    if panels and n in panels_n_list:
        ax = axs_list[panels_n_list.index(n)]
    m = n * int(round(tau_period / delta_tau, 1))
    t = float(times[n])
    t_str = np.format_float_positional(float(times[n]), precision=3, unique=False,
                                       fractional=False, trim='k')
    phase = 1 if t < t_1 else 2 if t < t_2 else 3
    N_x_current = np.count_nonzero(~np.isnan(data_dict["phi"][:, n]))
    # print(N_x_current)
    i_min, i_max = (N_x + 1 - N_x_current, N_x + 1)
    coord_arr_n = coord_arr[i_min:i_max]
    # print(coord_arr_n)

    # Calculate quantities within boundary layer
    early_bl_quants = []
    lin_quants = []
    late_bl_quants = []
    late_bl_extra_quants = []
    quasi_steady_quants = []
    if fluid_flux:
        phi_f_early = phi_f_bl_early(coord_arr_n, t, phi_f0, Q_f, D_phi_early)
    else:
        phi_f_early = phi_f_bl_early_pressure_drop(coord_arr_n, t, phi_f0, nu,
                                                   Delta_p, D_phi_early_pd, epsilon_E)
    # phi_f_lin = phi_f_lin_elastic(coord_arr_n, t, Q_f, phi_f0, D_phi_early)
    # phi_f_late = phi_f_bl_late(coord_arr_n, Q_f, D_phi_late)
    # phi_f_late_extra = phi_f_bl_late_longer(coord_arr_n, Q_f, D_phi_late, phi_f0, nu)
    x_arr_qss, phi_f_quasi_steady, _, _ = phi_f_quasi_steady_late(coord_arr, t, t_E, E_min, params, fixed_domain)
    early_bl_quants.append(phi_f_early)
    # lin_quants.append(phi_f_lin)
    # late_bl_quants.append(phi_f_late)
    # late_bl_extra_quants.append(phi_f_late_extra)
    quasi_steady_quants.append(phi_f_quasi_steady)

    """
    Set up figure for the overall plot
    """
    if gif:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, (10 * nrows)/3), sharex=True)
        plt.subplots_adjust(wspace=1.0 * (ncols - 1))
        axs_list = [axs]
    print("Time:", t_str)

    # Set up the correct arrays for the timepoint
    for i, name in enumerate(short_quants):
        if gif:
            ax = axs_list[i]
        coord_arr_n = np.log(1 - coord_arr_n) if log else coord_arr_n
        early_bl_n = np.log(early_bl_quants[i]) if log else early_bl_quants[i]
        # lin_n = np.log(lin_quants[i]) if log else lin_quants[i]
        # late_bl_n = np.log(late_bl_quants[i]) if log else late_bl_quants[i]
        # late_bl_extra_n = np.log(late_bl_extra_quants[i] if log else late_bl_extra_quants[i])
        qss_n = np.log(quasi_steady_quants[i]) if log else quasi_steady_quants[i]
        fenics_arr_n = np.log(data_dict[name][:, n]) if log else data_dict[name][:, n]
        if gif or panels:
            if panels and n not in panels_n_list:
                continue
            # line_early_bl = ax.plot(coord_arr_n, early_bl_n, "--g", label=early_label)
            # line_late_bl = ax.plot(coord_arr_n, late_bl_n, "--r", label=late_label)
            line_fenics = ax.plot(coord_arr_n, fenics_arr_n[i_min:i_max], color="black", label=fenics_label)
            line_early = ax.plot(coord_arr_n, early_bl_n, color="darkgoldenrod", linestyle='--', label=early_label)
            line_qss = ax.plot(x_arr_qss, qss_n, color="deepskyblue", linestyle="--", label=quasi_steady_label)
            ax.set_xlabel(plot_coord_tex)
            ax.set_ylabel(latex_quants[i])
            if gif:
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(mins[i], maxes[i])
            ax.set_title(f"$t = {t_str}$")
        else:
            phi_f.f_fixed = fenics_arr_n
            line_early_bl = ax.plot(coord_arr_n, early_bl_n, "--r",
                                    label=early_label if n == 0 else None)
            # line_lin = ax.plot(coord_arr_n, lin_n, color="darkviolet",
            #                    label=lin_label if n == 0 else None)
            # line_late_bl = ax.plot(coord_arr_n, late_bl_n, "--r",
            #                        label=late_label if n == 0 else None)
            # line_late_bl_extra = ax.plot(coord_arr_n, late_bl_extra_n, linestyle='--',
            #                              color='goldenrod', label=late_extra_label if n == 0 else None)
            line_qss = ax.plot(x_arr_qss, qss_n, color="firebrick", linestyle="--",
                               label=quasi_steady_label if n == 0 else None)
            line_fenics = phi_f.plot(norm, t, fixed_domain=True,
                                     label=fenics_label if n == 0 else None)
        if (panels and panels_n_list.index(n) == ncols * (nrows - 1)) or not panels:
            ax.legend()
        # if log:
        #     ax.set_xscale("log")
        #     ax.set_yscale("log")

    if gif:
        # Save figure
        frame_path = f"{plot_path}/frames/frame_{n}.png"
        fig.savefig(frame_path, bbox_inches="tight")
        image = Image.open(frame_path)
        images.append(image)
        os.remove(frame_path)

if gif:
    # Make the .gif and delete the frames directory
    images[0].save(f"{plot_path}/{short_quants[0]}_animated_bl{log_text}.gif", save_all=True,
                   append_images=images[1:], duration=100, loop=0)
    os.rmdir(f"{plot_path}/frames")
elif panels:
    fig.savefig(f"{plot_path}/{short_quants[0]}_bl_panels{log_text}.png", bbox_inches="tight")
else:
    # Set up the colorbar and save the figure
    Quantity.annotate_plots([phi_f], fig, norm, plot_coord_tex,
                            mins=[-2], maxes=maxes)
    fig.savefig(f"{plot_path}/{short_quants[0]}_bl_traces{log_text}.png", bbox_inches="tight")

"""
Plots for a and v
"""

fig_av, axs_av = plt.subplots(2, 1, sharex=True, figsize=(6, 6.66))
ax_a, ax_v = axs_av[0], axs_av[1]
response_df = pd.read_csv(f"{data_path}/_responses.csv", index_col=0)
times = response_df["Time"].to_numpy()

if read_a_v:
    a_df = pd.read_csv(f"{data_path}/a_analytic.csv", index_col=0)
    v_df = pd.read_csv(f"{data_path}/v_analytic.csv", index_col=0)
    a_fenics = a_df["Numeric"].to_numpy()
    v_fenics = v_df["Numeric"].to_numpy()
    a_early = a_df["Early"].to_numpy()
    v_early = v_df["Early"].to_numpy()
    a_qss_arr = a_df["QSS"].to_numpy()
    v_qss_arr = v_df["QSS"].to_numpy()
else:
    a_fenics = response_df["a"].to_numpy()
    v_fenics = response_df["v"].to_numpy()
    a_early = 2 * Delta_p * np.sqrt(times / (np.pi * D_phi_early_pd * epsilon_E))
    v_early = 1 / np.sqrt(np.pi * D_phi_early_pd * times / epsilon_E)

    a_qss_list, v_qss_list = [], []
    for t in times:
        _, _, a_qss, v_qss = phi_f_quasi_steady_late(coord_arr, t, t_E, E_min, params, fixed_domain)
        a_qss_list.append(a_qss)
        v_qss_list.append(v_qss)
    a_qss_arr, v_qss_arr = np.array(a_qss_list), np.array(v_qss_list)

# Plotting for a
ax_a.plot(times[1:], a_fenics[1:], color="black", label="Numerical")
ax_a.plot(times[1:], a_early[1:], linestyle="--", color="darkgoldenrod", label="Early time")
ax_a.plot(times[1:], a_qss_arr[1:], linestyle="--", color="deepskyblue", label="Quasi-steady")
ax_a.set_xscale("log")
ax_a.set_ylabel("$a$")
ax_a.set_ylim(-0.05, 0.3)
ax_a.legend()

a_df = pd.DataFrame({"Numeric": a_fenics, "Early": a_early, "QSS": a_qss_arr})
a_df.to_csv(f"{data_path}/a_analytic.csv")

# Plotting for v
ax_v.plot(times[1:], v_fenics[1:], color="black", label="Numerical")
ax_v.plot(times[1:], v_early[1:], linestyle="--", color="darkgoldenrod", label="Early time")
ax_v.plot(times[1:], v_qss_arr[1:], linestyle="--", color="deepskyblue", label="Quasi-steady")
ax_v.set_xscale("log")
ax_v.set_yscale("log")
ax_v.set_xlabel("$t$")
ax_v.set_ylabel("$v$")
ax_v.set_ylim(0.6, 300)
ax_v.legend()

v_df = pd.DataFrame({"Numeric": v_fenics, "Early": v_early, "QSS": v_qss_arr})
v_df.to_csv(f"{data_path}/v_analytic.csv")

fig_av.savefig(f"{plot_path}/{short_quants[0]}_bl_av{log_text}.png", bbox_inches="tight")

