"""
This Python code solves the following nonlinear, nondimensional general system for the
porosity, Young's modulus and solute concentration

        \\frac{t_{\\phi}}{[t]}\\frac{D^{f}\\phi_{f}}{Dt} = \\phi_{f}\\frac{\\p}{\\p x}
        \\left[k_{e}(\\phi_{f})\\frac{\\p}{\\p x}\\left(E\\sigma_{e}(\\phi_{f})\\right)\\right],
        \\frac{t_{E}}{[t]}\\frac{D^{s}E}{Dt} = -cE,
        \\frac{t_{c}}{[t]}\\phi_{f}\\frac{D^{f}c}{Dt} = \\frac{\\p}{\\p x}\\left(
        \\phi_{f}\\frac{\\p c}{\\p x}\\right),

where the operators D^{f} and D^{s} are the material derivatives for the fluid and solid
fractions respectively. These operators are dependent on E, k_{e}, \\sigma_{e} (both given
functions of the porosity) and v(t), which is a prescribed phase-averaged velocity.

The initial conditions are at t = 0:

        \\phi_{f} = \\phi_{f,0} (= const.), E = E_{0}(x), c = c_{0}(x),

with boundary conditions (on a domain [a(t), 1] with left moving boundary):

        v_s = \\frac{t_{v_{s}}}{[t]}\\dot{a}(t) at x = a(t), v_s = 0 at x = 1,
        c = 1 at x = a(t), \\frac{1}{t_{c}}\\frac{\\p c}{\\p x} - \\frac{1}{t_{v_{f}}}cv_{f} = 0 at x = 1,

The moving boundary can be determined by the following implicit relation:

        a(t) = \\phi_{f,0} - \\int_{a(t)}^{1}\\phi_{f}(x, t)dx,

given a known profile for \\phi_{f} at the previous timestep (in the numerical scheme).
We will also change coordinates onto a fixed domain (see details below).

The timescales are:

t_{\\phi} = \\frac{\\mu L^{2}}{k_{0}E^{*}},
t_{v_{i}} = \\frac{L}{v_{i}^{*}}, (v_{i} = v, v_{f} or v_{s}),
t_{E} = \\frac{1}{\\beta_{E}c^{*}},
t_{c} = \\frac{L^{2}}{\\mathcal{D}_{m}}.
"""
import os
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
trial = "nondim_initial"
sub_trial = "v_0_1"
param_file = open(f"resources/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

# Whether we'll plot on a fixed domain or not
fixed_domain = False
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"

"""
Define model parameters
"""

# Length of domain, L
L = Constant(params["phys"]["L"])

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(params["ics"]["phi_f"])

# Diffusive parameter for the solute concentration
D_m = Constant(params["phys"]["D_m"])

# Minimum (nondimensional) value of E (can be thought of as
# fraction of original E)
E_min = Constant(params["phys"]["E_min"])

# Poisson ratio and viscosity
nu = Constant(params["phys"]["nu"])
mu = Constant(params["phys"]["mu"])

# Permeability scale
k_0 = Constant(params["scales"]["k"])

# Solute concentration, Young's modulus and velocity scales
c_star = Constant(params["scales"]["c"])
E_star = Constant(params["scales"]["E"])
v_star = Constant(params["scales"]["v"])

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
t_c = L ** 2 / D_m

x = Expression('x[0]', degree=1)


def nums(*constants: Constant):
    return tuple([constant(0.0) for constant in constants])


phi_f0_num, nu_num = nums(phi_f0, nu)
t_v_num = nums(t_v)
t_phi_num, t_c_num = nums(t_phi, t_c)


# Setting up the moving boundary
a_list = []

"""
Computational parameters
"""

# Number of mesh points
N_x = params["comp"]["N_x"]

# Imposed phase-averaged velocity
vt = Expression(params["v"]["expr"],
                degree=1)

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the xi coodinates
xi = SpatialCoordinate(mesh)[0]
xi_arr = np.linspace(0, 1, N_x + 1)

# Set up function space
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
P0 = FiniteElement("R", mesh.ufl_cell(), 0)

# For vars phi_f, E, c, u_s, a
element = MixedElement([P1, P1, P1, P1, P0])
V = FunctionSpace(mesh, element)

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


def xi_t_to_x_t(_mesh: Mesh, _a: float, *_quantities: Quantity):
    """Change the quantity from (xi, t) coordinates to (x, t) where
    \\xi = 1 - \\frac{1 - x}{1 - a(t)}. We also fit onto the new mesh, which
    will involve some interpolation.

    :param _mesh: The mesh for the domain.
    :param _a: The moving boundary a(t).
    :param _quantities: A tuple of quantities in (xi, t) coordinates (np array).
    :return: The new tuple of arrays in (x, t) coordinates.
    """
    f_part_list = []
    for quantity in _quantities:
        _, _f = fenics_to_numpy(_mesh, quantity.f)
        # How many points there are in the (x, t) domain
        N_part = N_x - int(_a * N_x)
        # Shrink region to [0, 1] (using transformation) and interpolate onto
        # only the left part of the grid
        f_part = np.interp(np.linspace(0, 1, N_part + 1),
                           np.linspace(0, 1, N_x + 1),
                           _f)
        # Fill the left of array with NaNs if domain has been compressed
        if N_x - N_part >= 0:
            f_part = np.concatenate([np.full(N_x - N_part, np.nan), f_part])
            quantity.mesh_fixed = np.linspace(0, 1, N_x + 1)
        # Else, if domain has expanded, we must change the mesh
        else:
            dx = 1 / N_x
            N_neg = N_part - N_x
            mesh_fixed = np.linspace(- N_neg * dx, 1, N_x + N_neg + 1)
            # Update the fixed mesh of the quantity
            quantity.mesh_fixed = mesh_fixed

        f_part_list.append(f_part)

    return tuple(f_part_list)


"""
Define the solutions phi_f, E and c
"""

phi_f = Quantity("$\\phi_{f}$", "Blues", 0, mesh)
E = Quantity("$E$", "Purples", 1, mesh)
c = Quantity("$c$", "Reds", 2, mesh)
u_s = Quantity("$u_s$", "Greens", 3, mesh)

# Set up the functions from the joint space
v_phi, v_E, v_c, v_us, v_a = TestFunctions(V)

# Define the initial conditions
# w_0 = Expression(('phi_f0', params["ics"]["E"], params["ics"]["c"],
#                   params["ics"]["u_s"], 'a_0'),
#                  degree=1, phi_f0=phi_f0, a_0=a_list[0], E_min=E_min)
# w_old = project(w_0, V)


w = Function(V)
w_phi, w_E, w_c, w_us, a = split(w)
# phi_old, E_old, c_old, u_s_old, a_old = split(w_old)
# phi_f.f, E.f, c.f, u_s.f, a_f = w_old.split(deepcopy=True)
phi_f.set_sym_functions(w_phi, v_phi, w_phi)
E.set_sym_functions(w_E, v_E, w_E)
c.set_sym_functions(w_c, v_c, w_c)
u_s.set_sym_functions(w_us, v_us, w_us)

# Define also the known functions k_{e} and \\sigma_{e} (of porosity)


def compute_k_e(_phi_f, _phi_f0):
    """Computes k_e as a function of the porosity.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :return: The effective permeability.
    """
    numerator = (1 - _phi_f0) * (_phi_f ** 2)
    denominator = (_phi_f0 ** 3) * (1 - _phi_f)
    return numerator / denominator


def compute_sigma_e(_phi_f, _phi_f0, _nu):
    """Computes sigma_e as a function of the porosity.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :param _nu: Poisson's ratio.
    :return: The effective stress.
    """
    term1 = (1 - _phi_f0) ** 2 / (1 - _phi_f)
    term2 = - 2 * _nu * (1 - _phi_f0)
    term3 = - (1 - 2 * _nu) * (1 - _phi_f)
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    return (term1 + term2 + term3) / denominator


def compute_dsigma_e_dphi(_phi_f, _phi_f0, _nu):
    """Computes the derivative of sigma_e as a function of the porosity.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :param _nu: Poisson's ratio.
    :return: The derivative of the effective stress.
    """
    term1 = (1 - _phi_f0) ** 2 / (1 - _phi_f) ** 2
    term2 = 1 - 2 * _nu
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    return (term1 + term2) / denominator


k_e = Quantity("$k_{e}(\\phi_{f})$", "GnBu", 4, mesh)
sigma_e = Quantity("$\\sigma_{e}(\\phi_{f})$", "YlOrBr", 5, mesh)

"""
Define the Dirichlet boundary conditions
"""


# Define a function for the left boundary; this function
# just needs to return the value true when xi is close to
# the boundary 0
def left(xi):
    return near(xi[0], 0)


# Define a function for the right boundary; this function
# just needs to return the value true when xi is close to
# the boundary 1
def right(xi):
    return near(xi[0], 1)


# Define the boundary conditions at the left and right
bcs = []

if "c_left" in params["bcs"]:
    bc_left_c = DirichletBC(V.sub(2), params["bcs"]["c_left"], left)
    c.add_bc(bc_left_c)
    bcs.append(bc_left_c)
if "u_s_right" in params["bcs"]:
    bc_right_us = DirichletBC(V.sub(3), params["bcs"]["u_s_right"], right)
    u_s.add_bc(bc_right_us)
    bcs.append(bc_right_us)

bc_left_phi = DirichletBC(V.sub(0), 0.01, right)
phi_f.add_bc(bc_left_phi)
bcs.append(bc_left_phi)

param_file.close()


# Define our fluid and solid velocities on the right
# qt = 0
# _, v_s0_array = get_vs_R(mesh, phi_f1_R.f, x_c[0], phi_f0, qt, D_phi)
# _, v_f0_array = get_vf_R(mesh, v_s0_array, phi_f0, qt)
# v_s0_R = Quantity("$v_{s,0}^{R}$", "Oranges", 6, mesh)
# v_f0_R = Quantity("$v_{f,0}^{R}$", "PuRd", 7, mesh)
# v_s0_R.f = v_s0_array
# v_f0_R.f = v_f0_array


def get_vs_from_E_phi(_mesh, _phi_f, _E, _a, _phi_f0, _nu, 
                      _t_v, _t_vs, _t_phi):
    _, vt_arr = fenics_to_numpy(_mesh, vt)
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, E_arr = fenics_to_numpy(_mesh, _E)
    k_e_arr = compute_k_e(phi_f_arr, _phi_f0)
    sigma_e_arr = compute_sigma_e(phi_f_arr, _phi_f0, _nu)
    return (vt_arr / _t_v +
            phi_f_arr * k_e_arr * np.gradient(E_arr * sigma_e_arr, xi_arr)
            / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs
    # return (phi_f_arr * k_e_arr * np.gradient(E_arr * sigma_e_arr, xi_arr)
    #         / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs


v_s_ = Quantity("$v_{s}$", "YlOrBr", 7, mesh)
quantities = [phi_f, E, c, u_s, v_s_]


"""
Set up figure for the overall plot
"""
# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
Quantity.set_axs(quantities, axs)
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

"""
Plot the initial curves and save all our data
"""
# saving = [True, True, True, True]
# short_quants = ["phi", "E", "c", "u_s"]
# saving = [True, True, True, True, True]
saving = [False] * 5

for quantity in quantities:
    quantity.initialise_dataframe(xi_arr)

k_e.g = compute_k_e(phi_f.g, phi_f0)
sigma_e.g = compute_sigma_e(phi_f.g, phi_f0, nu)

"""
Define the weak form
"""

# Weak form for the phi equation
Fun_phi = ((t_phi / t_v * vt * phi_f.g - k_e.g * (E.g * sigma_e.g).dx(0) / (1 - a) * phi_f.g)
           * phi_f.v_0.dx(0) * dx) - t_phi / t_v * vt * phi_f.v_0 * ds
# Fun_phi = (phi_f.g - phi_f0) * phi_f.v * dx

# Weak form for the E equation
Fun_E = (E.g - E_min) * E.v_0 * dx

# Weak form for the c equation
Fun_c = (t_c / t_v * vt * c.g - phi_f.g / (1 - a) * c.g.dx(0)) * c.v_0.dx(0) * dx

# Weak form for the displacement
Fun_us = ((u_s.g.dx(0) * u_s.v_0 -
           (phi_f.g - phi_f0) * (1 - a) / (1 - phi_f0) * u_s.v_0) * dx)

# Weak form for the moving boundary
Fun_a = (phi_f.g - 1 + (1 - phi_f0) / (1 - a)) * v_a * dx

# Combining the weak forms
Fun = Fun_phi + Fun_E + Fun_c + Fun_us + Fun_a


# Define the Jacobian, problem and solver
jacobian = derivative(Fun, w)
problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
solver = NonlinearVariationalSolver(problem)

"""
Loop over time steps and solve
"""

# Solve
solver.solve()
phi_f.f, E.f, c.f, u_s_new, a_f = w.split(deepcopy=True)
v_s_.f = get_vs_from_E_phi(mesh, phi_f.f, E.f, a_f(0.0), phi_f0_num, nu_num,
                           t_v_num, t_v_num, t_phi_num)
a_list.append(a_f(0.0))

# Change coordinates onto the fixed domain for plotting
(phi_f.f_fixed, E.f_fixed, c.f_fixed,
 u_s.f_fixed, v_s_.f_fixed) = xi_t_to_x_t(mesh, a_list[-1], phi_f, E,
                                          c, u_s, v_s_)
# plot at the current timepoint
Quantity.plot_quantities(quantities, norm, 0.0, saving,
                         fixed_domain=fixed_domain)
u_s.f = u_s_new

# Save all relevant quantities
short_quants = ["phi", "E", "c", "u_s", "v_s"]
data_path = f"resources/{trial}/{sub_trial}/data"
if any(saving) and not os.path.isdir(data_path):
    os.makedirs(data_path)
file_names = [f"{data_path}/{q}.csv" for q in short_quants]
Quantity.write_to_csv(quantities, saving, file_names)

"""
Colourbars
"""

# phi_f
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f.cmap),
             orientation='vertical',
             label='$t$', ax=phi_f.ax)
phi_f.label_plot(x_label=plot_coord_tex, title="Porosity")

# E
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=E.cmap),
             orientation='vertical',
             label='$t$', ax=E.ax)
E.label_plot(x_label=plot_coord_tex, title="Young's modulus")

# c
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c.cmap),
             orientation='vertical',
             label='$t$', ax=c.ax)
c.label_plot(x_label=plot_coord_tex, title="Solute concentration")

# u_s
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s.cmap),
             orientation='vertical',
             label='$t$', ax=u_s.ax)
u_s.label_plot(x_label=plot_coord_tex, title="Displacement")

# diff
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_s_.cmap),
             orientation='vertical',
             label='$t$', ax=v_s_.ax)
v_s_.label_plot(x_label=plot_coord_tex, title="Difference")

# Remove titles
for ax in axs:
    ax.set_title("")

# Check plot directory exists
plot_path = f"resources/{trial}/{sub_trial}/plots"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Save figure
fig.savefig(f"{plot_path}/time_traces_{plot_coord}.png", bbox_inches="tight")

# Create figure for the left boundary over time
# fig_a, ax_a = plt.subplots()
# times = np.linspace(0, N_time * delta_t, N_time + 1)
# a_expected = times * params["v"]["v_final"] * params["scales"]["t"] / t_v_num
# ax_a.plot(np.array(a_list), times,
#           color='darkviolet', label='$a(t)$')
# ax_a.plot(np.array(v_s_0_list), times,
#           color='darkviolet', label='$v_s(0)$')
# ax_a.set_xlabel("Left boundary")
# ax_a.set_ylabel("Time")
# ax_a.legend()
# ax_a.set_xlim(min(a_list), max(a_list))
# fig_a.savefig(f"{plot_path}/left_bdry.png", bbox_inches="tight")
