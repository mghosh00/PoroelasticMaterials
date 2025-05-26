"""
This Python code solves the following linear, nondimensional equation for the porosity
and left boundary only

        \\diffp{\\phi_{f,1}}{t} = \\mathcal{D}_{\\phi}\\diffp[2]{\\phi_{f,1}}{x},

The initial conditions are at t = 0:

        \\phi_{f,1} = 0, a_{1} = 0,

with boundary conditions (on a domain [\\epsilon a_{1}(t), 1] with left moving boundary):

        \\phi_{f,1} = 0    on x = 0
        Q_{f,1} + \\frac{\\mathcal{D_{\\phi}}}{1 - \\phi_{f,0}}\\diffp{\\phi_{f,1}}{x} = 0    on x = 0,
        Q_{f,1} + \\dot{a_{1}} + \\frac{\\mathcal{D_{\\phi}}}{1 - \\phi_{f,0}}\\diffp{\\phi_{f,1}}{x} = 0    on x = 1,

The moving boundary can be determined by the following implicit relation:

        a_{1}(t) = \\frac{1}{1 - \\phi_{f,0}}\\int_{0}^{1}\\phi_{f,1}dx,

where we note here that this equation can replace one of the flux boundary conditions.
We have linearised onto a fixed domain.

Note that Q_{f,1}(t) is a prescribed function of time and

\\mathcal{D_{\\phi}} = \\frac{E_{0}(1 - \\nu)}{\\gamma(1 + \\nu)(1 - 2\\nu)}
"""
import os
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
trial = "linear"
sub_trial = "const_Q"
param_file = open(f"resources/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

num_quants = 3

# Whether to save data or not
saving = [True] * num_quants

"""
Computational parameters
"""

# Size of time step
delta_t = params["comp"]["delta_t"]

# Number of time steps
N_time = params["comp"]["N_time"]

# Number of mesh points
N_x = params["comp"]["N_x"]

# The frequency of plotting
# num_lines = 20
num_lines = N_time
plotting_freq = int(N_time / num_lines)

"""
Define model parameters
"""

# Timescale, [t]
t_sc = Constant(params["scales"]["t"])

# Length of domain, L
L = Constant(params["phys"]["L"])

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(params["ics"]["phi_f"])

# Minimum (nondimensional) value of E (can be thought of as
# fraction of original E)
E_min = Constant(params["phys"]["E_min"])

# Poisson ratio and viscosity
nu = Constant(params["phys"]["nu"])
mu = Constant(params["phys"]["mu"])

# Permeability scale
k_0 = Constant(params["scales"]["k"])

# Solute concentration, Young's modulus and velocity scales
E_star = Constant(params["scales"]["E"])
v_star = Constant(params["scales"]["v"])

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
gamma = t_phi / t_v
D_phi = E_min * (1 - nu) / (gamma * (1 + nu) * (1 - 2 * nu))

def nums(*constants: Constant):
    return tuple([constant(0.0) for constant in constants])


t_sc_num, phi_f0_num, nu_num = nums(t_sc, phi_f0, nu)
t_v_num, t_phi_num, gamma_num, D_phi_num = nums(t_v, t_phi, gamma, D_phi)


# Setting up the moving boundary
a_list = [params["ics"]["a"]]


"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the x coordinate
x = SpatialCoordinate(mesh)[0]
x_arr = np.linspace(0, 1, N_x + 1)

# Set up function space
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P0 = FiniteElement("R", mesh.ufl_cell(), 0)

# For vars phi_f11, v_1, a_0
element = MixedElement([P1, P1, P0])
V = FunctionSpace(mesh, element)

# Imposed fluid flux
Q_f = Expression(params["Q"]["expr"],
                 degree=1, t=0.0, delta_t=delta_t, N_time=N_time, domain=mesh)
"""
Function to change spatial coordinates and get to the correct mesh.
"""


def fenics_to_numpy(_mesh: Mesh, f: Function):
    """Converts a FEniCS function to numpy

    :param _mesh: The mesh
    :param f: The function
    :return: The numpy arrays for the coordinates and function
    """
    # If numpy arrays are passed, just return them back
    mesh_array = (_mesh if isinstance(_mesh, np.ndarray)
                  else np.array(_mesh.coordinates()))
    f_array = (f if isinstance(f, np.ndarray)
               else f.compute_vertex_values(_mesh))
    return mesh_array, f_array


"""
Define the solutions phi_f1, v
"""

phi_f1 = Quantity("$\\phi_{f,1}$", "Blues", 0, mesh)
v_0 = Quantity("$v_0$", "RdPu", 1, mesh)

# Set up the functions from the joint space
v_phi, v_v, v_a = TestFunctions(V)


def retrieve_ic(_xi, data_array: np.array):
    """If we wish to set up initial conditions with a numpy
    array, this function will be used in the interpolation.

    :param _xi: The FEniCS coordinate
    :param data_array: The data array
    :return: The value of the function at the specific coordinate
    """
    _N_x = len(data_array)
    index = int(float(_xi) * _N_x)
    return data_array[index]


short_quants = ["phi", "v", "v_s"]
data_path = f"resources/{trial}/{sub_trial}/data"

# Define the initial conditions
phi_f0_ic = 'phi_f0 - gamma * (1 - phi_f0) * (1 + nu) * (1 - 2 * nu) / (1 - nu) * x[0]'
w_0 = Expression(('0.0', '0.0', 'a_0'),
                 degree=1, phi_f0=phi_f0, a_0=a_list[0])
w_old = project(w_0, V)

# w_old = Function(V)
w = Function(V)
w_phi, w_v, a_0 = split(w)
phi_old, v_old, a_0_old = split(w_old)

phi_f1.f, v_0.f, a_f = w_old.split(deepcopy=True)
phi_f1.set_sym_functions(w_phi, v_phi, phi_old)
v_0.set_sym_functions(w_v, v_v, v_old)


"""
Define the Dirichlet boundary conditions
"""


boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)


# Define a function for the left boundary; this function
# just needs to return the value true when x is close to
# the boundary 0
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1) and on_boundary


# Define the boundary conditions at the left and right
left = Left()
left.mark(boundary_markers, 1)
right = Right()
right.mark(boundary_markers, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
bcs = []

phi_l = params["bcs"]["phi_left"]
bc_left_phi = DirichletBC(V.sub(0), phi_l, boundary_markers, 1)
bc_right_phi = DirichletBC(V.sub(0), phi_l, boundary_markers, 2)
bcs.append(bc_left_phi)
# bcs.append(bc_right_phi)

param_file.close()


def get_vs_from_E_phi(_mesh, _phi_f1, _a_list, _phi_f0, _D_phi):
    _, Q_f_arr = fenics_to_numpy(_mesh, Q_f)
    _, phi_f1_arr = fenics_to_numpy(_mesh, _phi_f1)
    da_dt_val = (_a_list[-1] - _a_list[-2]) / delta_t
    print(da_dt_val)
    _a = _a_list[-1]
    dphi_dx = np.gradient(phi_f1_arr, x_arr)
    _v_s = Q_f_arr + da_dt_val + _D_phi / (1 - _phi_f0) * dphi_dx
    return _v_s


# This quantity is just for plotting purposes
v_s_ = Quantity("$v_{s,0}$", "YlOrBr", 2, mesh)
# We don't know the initial array v_s
v_s_.f = np.full(N_x + 1, np.nan)
quantities = [phi_f1, v_s_, v_0]


"""
Set up figure for the overall plot
"""
nrows, ncols = 3, 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
plt.subplots_adjust(wspace=1.0 * (ncols - 1))
axs_list = [axs[i] for i in range(nrows)]
Quantity.set_axs(quantities, axs_list)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves and save all our data
"""

for quantity in quantities:
    quantity.initialise_dataframe(x_arr)

Quantity.plot_quantities(quantities, norm, 0.0, saving)

# define the time derivatives
dphi_dt = (phi_f1.u - phi_f1.u_old) / delta_t
da_dt = (a_0 - a_0_old) / delta_t

"""
Define the weak form
"""

Fun_phi = (dphi_dt * phi_f1.v * dx + D_phi * phi_f1.u.dx(0) * phi_f1.v.dx(0) * dx +
           (1 - phi_f0) * v_0.u * phi_f1.v * ds(2) - (1 - phi_f0) * Q_f * phi_f1.v * ds(1))

# Weak form for the phase-averaged velocity
Fun_v = (((Q_f - v_0.u) / t_v + da_dt / t_sc) * v_0.v * dx)

# Weak form for the moving boundary
Fun_a = (phi_f1.u + (1 - phi_f0) * a_0) * v_a * dx
# Fun_a = a_0 * v_a * dx

# Combining the weak forms
Fun = Fun_phi + Fun_v + Fun_a


# Define the Jacobian, problem and solver
jacobian = derivative(Fun, w)

"""
Loop over time steps and solve
"""
Q_f_list = [float(Q_f(0.0))]
for n in range(N_time):
    problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
    solver = NonlinearVariationalSolver(problem)
    print("Time:", np.round(n * delta_t, 3))

    # Solve
    solver.solve()
    phi_f1.f, v_0.f, a_f = w.split(deepcopy=True)
    _, phi_f1_arr = fenics_to_numpy(mesh, phi_f1.f)
    a_list.append(a_f(0.0))
    v_s_.f = get_vs_from_E_phi(mesh, phi_f1.f, a_list, phi_f0_num, D_phi_num)
    w_old.assign(w)

    # plot at the current timepoint if needed
    if (n + 1) % plotting_freq == 0:
        Quantity.plot_quantities(quantities, norm, (n + 1) * delta_t, saving,
                                 fixed_domain=False)

    # Update some variables
    Q_f.t = (n + 1) * delta_t
    Q_f_list.append(float(Q_f(0.0)))

# Save all relevant quantities
if any(saving) and not os.path.isdir(data_path):
    os.makedirs(data_path)
file_names = [f"{data_path}/{q}.csv" for q in short_quants]
Quantity.write_to_csv(quantities, saving, file_names)

# Set up the colorbars and label the plots
Quantity.annotate_plots(quantities, fig, norm, "$x$")

# Check plot directory exists
plot_path = f"resources/{trial}/{sub_trial}/plots"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Save figure
fig.savefig(f"{plot_path}/time_traces.png", bbox_inches="tight")


# Create figure for the imposed velocity and left boundary over time
def Q_a_v_plot(_t_arr: np.array, _a_arr: np.array, _Q_f_arr: np.array,
               _plot_path: str, _title: str):
    """Creates the figure for the fluid flux, left boundary and volume-averaged flux.

    :param _t_arr: The time array.
    :param _a_arr: The left boundary array.
    :param _Q_f_arr: The fluid flux array.
    :param _plot_path: The directory for the plot.
    :param _title: The title of the plot.
    """
    fig_Q_a_v, axs_Q_a_v = plt.subplots(nrows=3, ncols=1, figsize=(8, 30/3), sharex=True)
    ax_Q, ax_a, ax_v = axs_Q_a_v
    ax_Q.plot(_t_arr, _Q_f_arr,
              color='forestgreen', label='$Q_f(t)$')
    ax_Q.set_ylabel("Imposed velocity")
    ax_Q.legend()
    ax_a.plot(_t_arr, _a_arr,
              color='darkviolet', label='$a(t)$')
    ax_a.set_ylabel("Left boundary")
    ax_a.legend()
    ax_v.plot(_t_arr, _Q_f_arr + np.gradient(_a_arr, _t_arr),
              color='darkgoldenrod', label='$v(t)$')
    ax_v.set_xlabel("Time")
    ax_v.set_ylabel("Phase-averaged velocity")
    ax_v.legend()
    fig_Q_a_v.savefig(f"{_plot_path}/{_title}.png", bbox_inches="tight")


Q_a_v_plot(times, np.array(a_list), np.array(Q_f_list), plot_path, "Q_a_v")
# Here we calculate the analytic solutions
# ASSUMPTIONS
# We have zero solid stress on the right boundary
# The imposed fluid flux Q_f is constant in time


def phi_f1_analytic(_x_arr: np.array, _t: float, _Q_f: float,
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
    _phi_f1 = (1 - _phi_f0) / _D_phi * _Q_f * (1 - _x_arr)
    for m in range(n_terms):
        factor = - 8 / ((2 * m + 1) * np.pi) ** 2 * (1 - _phi_f0) / _D_phi * _Q_f
        summand = np.cos((m + 1 / 2) * np.pi * _x_arr) * np.exp(- ((m + 1 / 2) * np.pi) ** 2 * _D_phi * _t)
        _phi_f1 += factor * summand
    return _phi_f1


def a_analytic(_t: float, _Q_f: float, _D_phi: float, n_terms: int = 1000):
    """Calculates the analytic expression for the left boundary when we have
    zero solid stress on the RIGHT boundary. Q_f must also be constant in time.

    :param _t: The current timepoint.
    :param _Q_f: The imposed fluid flux.
    :param _D_phi: The diffusion coefficient in the porosity equation.
    :param n_terms: The number of terms in the series.
    :return: The left boundary at the current timepoint.
    """
    _a = 0.0
    for m in range(n_terms):
        factor = - 16 * _Q_f / _D_phi * (-1) ** m / ((2 * m + 1) * np.pi) ** 3
        summand = 1 - np.exp(-((m + 1 / 2) * np.pi) ** 2 * _D_phi * _t)
        _a += factor * summand
    return _a


def v_s_analytic(_phi_f1: np.array, _x_arr: np.array, _da_dt: float, _Q_f: float,
                 _phi_f0: float, _D_phi: float):
    """Calculates the analytic expression for v_s given the porosity and left boundary.
    Note the same caveats as for phi_f1 and a.

    :param _phi_f1: The current porosity array.
    :param _x_arr: The spatial coordinate array.
    :param _da_dt: The current velocity of the left boundary.
    :param _Q_f: The imposed fluid flux.
    :param _phi_f0: The initial porosity.
    :param _D_phi: The diffusion coefficient for the porosity equation.
    :return: The solid velocity at the current timepoint.
    """
    dphi_dx = np.gradient(_phi_f1, _x_arr)
    return _Q_f + _da_dt + _D_phi / (1 - _phi_f0) * dphi_dx


quantities_an = [phi_f1, v_s_]
nrows, ncols = 2, 1
fig_an, axs_an = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
plt.subplots_adjust(wspace=1.0 * (ncols - 1))
axs_list_an = [axs_an[i] for i in range(nrows)]
Quantity.set_axs(quantities_an, axs_list_an)

Q_f_num = Q_f_list[0]

# Prepare with the left boundary
a_arr_an = np.array([a_analytic(t, Q_f_num, D_phi_num) for t in times])
da_dt_arr = np.gradient(a_arr_an, times)

# Calculate the analytic solutions at each timestep
for n in range(len(times)):
    t = n * delta_t
    phi_f1.f = phi_f1_analytic(x_arr, t, Q_f_num, phi_f0_num, D_phi_num)
    v_s_.f = v_s_analytic(phi_f1.f, x_arr, da_dt_arr[n], Q_f_num, phi_f0_num, D_phi_num)
    if n % plotting_freq == 0:
        Quantity.plot_quantities(quantities_an, norm, n * delta_t, [False, False],
                                 fixed_domain=False)

# Annotate the plots
Quantity.annotate_plots(quantities_an, fig_an, norm, "$x$")

# Save figure
fig_an.savefig(f"{plot_path}/time_traces_analytic.png", bbox_inches="tight")

# Finally plot Q, a and v
Q_a_v_plot(times, a_arr_an, np.array([Q_f_num] * (N_time + 1)), plot_path,
           "Q_a_v_analytic")
