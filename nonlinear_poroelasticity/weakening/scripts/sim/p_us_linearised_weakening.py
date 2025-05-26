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
sub_trial = "p_us_const_Q"
param_file = open(f"resources/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

num_quants = 4

# Whether to save data or not
saving = [False] * num_quants

"""
Computational parameters
"""

# Size of time step
delta_t = Constant(params["comp"]["delta_t"])

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

# Solute concentration and Young's modulus values
E_star = Constant(params["scales"]["E"])
v_star = Constant(params["scales"]["v"])

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
gamma = t_phi / t_v
D_phi = E_min * (1 - nu) / (gamma * (1 + nu) * (1 - 2 * nu))

def nums(*constants: Constant):
    return tuple([constant(0.0) for constant in constants])


delta_t_num, t_sc_num, phi_f0_num, nu_num = nums(delta_t, t_sc, phi_f0, nu)
t_v_num, t_phi_num, gamma_num, D_phi_num = nums(t_v, t_phi, gamma, D_phi)


# Setting up the moving boundary
a_list = [params["ics"]["a"]]
Delta_P_list = [params["ics"]["Delta_P"]]


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

# For vars u_s0, p_f0, a_0, Delta_P0
element = MixedElement([P1, P1, P0, P0])
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
Define the solutions u_s0, p_f0
"""

u_s0 = Quantity("$u_{s,0}$", "Greens", 0, mesh)
p_f0 = Quantity("$p_{f,0}$", "Oranges", 1, mesh)

# Set up the functions from the joint space
v_us, v_pf, v_a, v_Delta_P = TestFunctions(V)


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


short_quants = ["u_s", "p_f"]
data_path = f"resources/{trial}/{sub_trial}/data"

# Define the initial conditions
phi_f0_ic = 'phi_f0 - gamma * (1 - phi_f0) * (1 + nu) * (1 - 2 * nu) / (1 - nu) * x[0]'
w_0 = Expression(('0.0', '0.0', 'a_0', 'Delta_P_0'),
                 degree=1, a_0=a_list[0], Delta_P_0=Delta_P_list[0])
w_old = project(w_0, V)

# w_old = Function(V)
w = Function(V)
w_us, w_pf, a_0, Delta_P_0 = split(w)
us_old, pf_old, a_0_old, Delta_P_0_old = split(w_old)

u_s0.f, p_f0.f, a_f, Delta_P_f = w_old.split(deepcopy=True)
u_s0.set_sym_functions(w_us, v_us, us_old)
p_f0.set_sym_functions(w_pf, v_pf, pf_old)


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

bc_right_us = DirichletBC(V.sub(0), 0, boundary_markers, 2)
bc_right_pf = DirichletBC(V.sub(1), 0, boundary_markers, 2)
bcs.append(bc_right_us)
bcs.append(bc_right_pf)

param_file.close()
quantities = [u_s0, p_f0]


"""
Set up figure for the overall plot
"""
nrows, ncols = 2, 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
plt.subplots_adjust(wspace=1.0 * (ncols - 1))
axs_list = [axs[i] for i in range(nrows)]
Quantity.set_axs(quantities, axs_list)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t_num)

times = np.linspace(0, N_time * delta_t_num, N_time + 1)

"""
Plot the initial curves and save all our data
"""

for quantity in quantities:
    quantity.initialise_dataframe(x_arr)

Quantity.plot_quantities(quantities, norm, 0.0, saving)

# define the time derivatives
dus_dt = (u_s0.u - u_s0.u_old) / delta_t
dpf_dt = (p_f0.u - p_f0.u_old) / delta_t
da_dt = (a_0 - a_0_old) / delta_t
dDelta_P_dt = (Delta_P_0 - Delta_P_0_old) / delta_t

"""
Define the weak form
"""

# Weak form for the displacement
Fun_us = ((dus_dt - Q_f) * u_s0.v * dx + D_phi * u_s0.u.dx(0) * u_s0.v.dx(0) * dx
          + 0 * Delta_P_0 * u_s0.v * ds(1))

# Weak form for the pressure
Fun_pf = ((dpf_dt - dDelta_P_dt) * p_f0.v * dx + D_phi * p_f0.u.dx(0) * p_f0.v.dx(0) * dx
          - D_phi * (Q_f - da_dt) * p_f0.v * ds(1))

# Weak form for the moving boundary
Fun_a = (u_s0.u - a_0) * v_a * ds(1)

# Weak form for the pressure difference
Fun_Delta_P = (p_f0.u - Delta_P_0) * v_Delta_P * ds(1)

# Combining the weak forms
Fun = Fun_us + Fun_pf + Fun_a + Fun_Delta_P


# Define the Jacobian, problem and solver
jacobian = derivative(Fun, w)

"""
Loop over time steps and solve
"""
Q_f_list = [float(Q_f(0.0))]
for n in range(N_time):
    problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
    solver = NonlinearVariationalSolver(problem)
    print("Time:", np.round(n * delta_t_num, 3))

    # Solve
    solver.solve()
    u_s0.f, p_f0.f, a_f, Delta_P_f = w.split(deepcopy=True)
    a_list.append(a_f(0.0))
    Delta_P_list.append(Delta_P_f(0.0))
    w_old.assign(w)

    # plot at the current timepoint if needed
    if (n + 1) % plotting_freq == 0:
        Quantity.plot_quantities(quantities, norm, (n + 1) * delta_t_num, saving,
                                 fixed_domain=False)

    # Update some variables
    Q_f.t = (n + 1) * delta_t_num
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
def Q_a_v_DeltaP_plot(_t_arr: np.array, _a_arr: np.array, _Q_f_arr: np.array,
                      _Delta_P_arr: np.array, _plot_path: str, _title: str):
    """Creates the figure for the fluid flux, left boundary and volume-averaged flux.

    :param _t_arr: The time array.
    :param _a_arr: The left boundary array.
    :param _Q_f_arr: The fluid flux array.
    :param _Delta_P_arr: The pressure difference array.
    :param _plot_path: The directory for the plot.
    :param _title: The title of the plot.
    """
    _fig, _axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 40/3), sharex=True)
    ax_Q, ax_a, ax_v, ax_Delta_P = _axs
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
    ax_Delta_P.plot(_t_arr, _Delta_P_arr,
                    color='orange', label='$\\Delta P(t)$')
    ax_Delta_P.set_ylabel("Pressure difference")
    ax_Delta_P.legend()
    _fig.savefig(f"{_plot_path}/{_title}.png", bbox_inches="tight")


Q_a_v_DeltaP_plot(times, np.array(a_list), np.array(Q_f_list), np.array(Delta_P_list),
                  plot_path, "Q_a_v_DeltaP")
