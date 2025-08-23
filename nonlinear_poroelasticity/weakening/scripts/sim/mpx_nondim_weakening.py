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

If we wish to have a nonlinear timestep, we will set our real time t = t(\\tau), where
the function t(\\tau) depends on some parameter \\tau and is defined within the .json
parameter files. The array of \\tau values will be defined on an array of length
N_time with constant spacing delta_tau.
"""

import os
import sys

import numexpr as _ne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from ufl import *
import dolfinx.mesh as dmesh
import dolfinx.fem as fem
from multiphenicsx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from nonlinear_poroelasticity.weakening.scripts import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
parent = "phys"
trial = "cardiovascular_stent"
sub_trial = "thickness"
param_file = open(f"resources/{parent}/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

# Whether we'll plot on a fixed domain or not
fixed_domain = False
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"
num_quants = 8

# Whether to save data or not
saving = [True] * num_quants

"""
Computational parameters
"""

# Size of time step (if the time array has a linear spacing)
delta_tau = params["comp"]["delta_tau"]

# Number of time steps
N_time = params["comp"]["N_time"]

# Number of mesh points
N_x = params["comp"]["N_x"]

# Whether we plot log time or linear time (boolean)
log_time = True if "log_time" in params["comp"] else False

# The frequency of plotting
num_lines = N_time / 1
# num_lines = 50
plotting_freq = int(N_time / num_lines)

"""
Create the mesh
"""

mesh_ = dmesh.create_interval(MPI.COMM_WORLD, N_x, [0.0, 1.0])

"""
Define model parameters
"""

# Timescale, [t]
t_sc = fem.Constant(mesh_, params["scales"]["t"])

# Length of domain, L
L = fem.Constant(mesh_, params["phys"]["L"])

# Initial porosity, \\phi_{f,0}
phi_f0 = fem.Constant(mesh_, params["ics"]["phi_f"])

# Degradation parameter
beta_E = fem.Constant(mesh_, params["phys"]["beta_E"])

# Diffusive parameter for the solute concentration
D_m = fem.Constant(mesh_, params["phys"]["D_m"])

# Minimum (nondimensional) value of E (can be thought of as
# fraction of original E)
E_min = fem.Constant(mesh_, params["phys"]["E_min"])

# Poisson ratio and viscosity
nu = fem.Constant(mesh_, params["phys"]["nu"])
mu = fem.Constant(mesh_, params["phys"]["mu"])

# Permeability scale
k_0 = fem.Constant(mesh_, params["scales"]["k"])

# Solute concentration, Young's modulus and velocity scales
c_star = fem.Constant(mesh_, params["scales"]["c"])
E_star = fem.Constant(mesh_, params["scales"]["E"])
v_star = fem.Constant(mesh_, params["scales"]["v"])
v_f_star = fem.Constant(mesh_, params["scales"]["v_f"])
v_s_star = fem.Constant(mesh_, params["scales"]["v_s"])

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
t_v_f = L / v_f_star
t_v_s = L / v_s_star
t_E = 1 / (beta_E * c_star)
t_c = L ** 2 / D_m

c_plus, c_minus = 0, 0
if "c_plus" in params["bcs"]:
    c_plus = fem.Constant(mesh_, params["bcs"]["c_plus"])
if "c_minus" in params["bcs"]:
    c_minus = fem.Constant(mesh_, params["bcs"]["c_minus"])


def nums(*constants: fem.Constant):
    return tuple([constant(0.0) for constant in constants])


t_sc_num, phi_f0_num, nu_num = nums(t_sc, phi_f0, nu)
t_v_num, t_v_f_num, t_v_s_num = nums(t_v, t_v_f, t_v_s)
t_phi_num, t_E_num, t_c_num = nums(t_phi, t_E, t_c)


# Setting up the moving boundary
a_list = [params["ics"]["a"]]


# get the xi coodinates
xi = SpatialCoordinate(mesh_)[0]
xi_arr = np.linspace(0, 1, N_x + 1)

# Set up function space
# P1 = FiniteElement("CG", mesh_.ufl_cell(), 1)
# P0 = FiniteElement("DG", mesh_.ufl_cell(), 0)
#
# # For vars phi_f, E, c, sigma, u_s, p_f, v, a
# element = MixedElement([P1, P1, P1, P1, P1, P1, P1, P0])
# V = FunctionSpace(mesh_, element)
V1 = fem.functionspace(mesh_, ("CG", 1))
V0 = fem.functionspace(mesh_, ("DG", 0))
V = (V1, V1, V1, V1, V1, V1, V0, V0)


class CustomConstant:
    def __init__(self, expr: str, _val_dict: dict[str, float]):
        self._expr = expr
        self._val_dict = _val_dict
        self.c = fem.Constant(mesh_, 0.0)
        self.update_val(_val_dict)

    def update_val(self, new_val_dict: dict[str, float]):
        for _k in new_val_dict:
            if _k in self._val_dict:
                self._val_dict[_k] = new_val_dict[_k]
        val = _ne.evaluate(self._expr, local_dict=new_val_dict)
        self.c.value = val
        return val


# Time expression and time array for a simulation with changing timestep
t_tau = params["comp"]["t(tau)"] if "t(tau)" in params["comp"] else "tau"
t = CustomConstant(t_tau, {"tau": 0.0, "delta_tau": delta_tau, "N_time": N_time})
t_next = CustomConstant(t_tau, {"tau": delta_tau, "delta_tau": delta_tau, "N_time": N_time})
delta_t = CustomConstant(t_tau, {"tau": 0.0, "delta_tau": delta_tau, "N_time": N_time})
delta_t.t = t_next.c - t.c
t_final = CustomConstant(t_tau, {"tau": N_time * delta_tau, "delta_tau": delta_tau, "N_time": N_time}).c(0.0)

"""
Function to change spatial coordinates and get to the correct mesh.
"""


def fenics_to_numpy(_mesh, f):
    """Converts a FEniCS function to numpy

    :param _mesh: The mesh
    :param f: The function
    :return: The numpy arrays for the coordinates and function
    """
    # If numpy arrays are passed, just return them back
    mesh_array = (_mesh if isinstance(_mesh, np.ndarray)
                  else np.array(_mesh.geometry.x))
    f_array = (f if isinstance(f, np.ndarray)
               else f.x.array.copy())
    return mesh_array, f_array


def xi_t_to_x_t(_mesh, _a: float, *_quantities: Quantity):
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

phi_f = Quantity("$\\phi_{f}$", "Blues", 0, mesh_)
E = Quantity("$E$", "Purples", 1, mesh_)
c = Quantity("$c$", "Reds", 2, mesh_)
sigma = Quantity("$\\sigma_{xx}'$", "Greys", 3, mesh_)
u_s = Quantity("$u_s$", "Greens", 4, mesh_)
p_f = Quantity("$p_f$", "Oranges", 5, mesh_)
v = Quantity("$v$", "RdPu", 6, mesh_)
_quantities = [phi_f, E, c, sigma, u_s, p_f, v]

# Set up the functions from the joint space
test_fns = [TestFunction(V[i]) for i in range(len(V))]
v_phi, v_E, v_c, v_sigma, v_us, v_pf, v_v, v_a = tuple(test_fns)


def const_ic(val: float):
    """Creates a lambda expression for a constant initial condition

    :param val: The constant.
    :return: The lambda expression.
    """
    return lambda _x : val * np.ones_like(_x[0])


short_quants = ["phi", "E", "p_f", "sigma", "u_s", "v_s", "c", "v"]
# short_quants = ["phi", "E", "c", "sigma", "u_s", "p_f", "v_s"]
data_path = f"resources/{parent}/{trial}/{sub_trial}/data"

# Define the initial conditions
ics = [const_ic(phi_f0_num), const_ic(params["ics"]["E"]), 
       lambda _x: c_minus + (c_plus - c_minus) * _x[0],
       const_ic(0.0), const_ic(params["ics"]["u_s"]), const_ic(0.0), const_ic(0.0), const_ic(0.0)]

w_old_list = [fem.Function(V[i]) for i in range(len(V))]
for i in range(len(w_old_list)):
    w_old_list[i].interpolate(ics[i])

w_list = [TrialFunction(V[i]) for i in range(len(V))]
w_phi, w_E, w_c, w_sigma, w_us, w_pf, w_v, a = tuple(w_list)
phi_old, E_old, c_old, sigma_old, u_s_old, p_f_old, v_old, a_old = tuple(w_old_list)


def f_copy(w_list):
    return tuple([w_list[i].copy() for i in range(len(w_list))])


phi_f.f, E.f, c.f, sigma.f, u_s.f, p_f.f, v.f, a_f = f_copy(w_old_list)
phi_f.set_sym_functions(w_phi, v_phi, phi_old)
E.set_sym_functions(w_E, v_E, E_old)
sigma.set_sym_functions(w_sigma, v_sigma, sigma_old)
c.set_sym_functions(w_c, v_c, c_old)
u_s.set_sym_functions(w_us, v_us, u_s_old)
p_f.set_sym_functions(w_pf, v_pf, p_f_old)
v.set_sym_functions(w_v, v_v, v_old)
p = fenics_to_numpy(mesh_, phi_f.f)
# Define also the known functions k_{e} and \\sigma_{e} (of porosity)


def compute_k_e(_phi_f, _phi_f0):
    """Computes k_e as a function of the porosity.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :return: The effective permeability.
    """
    numerator = (1 - _phi_f0) * (_phi_f ** 2)
    denominator = pow(_phi_f0, 3) * (1 - _phi_f)
    return numerator / denominator


def compute_g(_phi_f, _phi_f0, _nu):
    """Computes g as a function of the porosity.

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


def compute_dg_dphi(_phi_f, _phi_f0, _nu):
    """Computes the derivative of g as a function of the porosity.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :param _nu: Poisson's ratio.
    :return: The effective stress.
    """
    term1 = (1 - _phi_f0) ** 2 / (1 - _phi_f) ** 2
    term2 = 1 - 2 * _nu
    denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
    return (term1 + term2) / denominator


def compute_phi_f(_sigma, _E, _phi_f0, _nu):
    """Computes the porosity, phi_f, as a function of the Terzaghi stress
    and Young's modulus (an inverse relation).

    :param _sigma: The Terzaghi stress.
    :param _E: The Young's modulus.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :return: The porosity in terms of the other variables.
    """
    b = (1 + _nu) * (1 - 2 * _nu) * _sigma / _E + 2 * _nu
    discriminant = b ** 2 + 4 * (1 - 2 * _nu)
    factor = (1 - _phi_f0) / (2 * (1 - 2 * _nu))
    return 1 - factor * (discriminant ** (1 / 2) - b)


"""
Define the Dirichlet boundary conditions
"""


def get_phi_bc(_mesh, _sigma_xx, _E, _phi_f0, _nu, _left=True):
    """Finds the value of phi_f at a point given sigma and E.

    :param _mesh: The mesh.
    :param _sigma_xx: The value of sigma' at a point x in the domain.
    :param _E: The full Young's modulus profile.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :param _left: Whether we are on the left or right of the domain.
    :return: The value of the porosity at x.
    """
    _, E_arr = fenics_to_numpy(_mesh, _E)
    E_point = float(E_arr[0]) if _left else float(E_arr[-1])
    b = 2 * (1 + _nu) * (1 - 2 * _nu) * _sigma_xx / E_point + 2 * _nu
    discriminant = b ** 2 + 4 * (1 - 2 * _nu)
    factor = (1 - _phi_f0) / (2 * (1 - 2 * _nu))
    return 1 - factor * (discriminant ** (1 / 2) - b)


# Define a function for the left boundary; this function
# just needs to return the value true when xi is close to
# the boundary 0
def left_boundary(_xi):
    return np.isclose(_xi[0], 0.0)


def right_boundary(_xi):
    return np.isclose(_xi[0], 1.0)


left_facets = dmesh.locate_entities_boundary(mesh_, mesh_.topology.dim - 1, left_boundary)
left_dofs_list = [fem.locate_dofs_topological(V[i], mesh_.topology.dim - 1, left_facets)
                  for i in range(len(V))]
right_facets = dmesh.locate_entities_boundary(mesh_, mesh_.topology.dim - 1, right_boundary)
right_dofs_list = [fem.locate_dofs_topological(V[i], mesh_.topology.dim - 1, right_facets)
                   for i in range(len(V))]
facet_indices = np.concatenate([left_facets, right_facets])
facet_values = np.concatenate([np.ones_like(left_facets), 2 * np.ones_like(right_facets)])
boundary_markers = dmesh.meshtags(mesh_, mesh_.topology.dim - 1, facet_indices, facet_values)

ds = Measure('ds', domain=mesh_, subdomain_data=boundary_markers)
bcs = []

sigma_l_num = params["bcs"]["sigma_left"]
sigma_l = fem.Constant(mesh_, sigma_l_num)
bc_left_phi = fem.dirichletbc(get_phi_bc(mesh_, sigma_l_num, E.f, phi_f0_num, nu_num), left_dofs_list[0], V[0])

bcs.append(bc_left_phi)
# Imposed fluid flux or pressure drop
imposed_flux = False
if "Q_f" in params and "Delta p" not in params["bcs"]:
    imposed_flux = True
    Q_f = CustomConstant(params["Q_f"]["expr"],
                         {"t": t.c(0.0), "delta_tau": delta_tau, "N_time": N_time})
elif "Delta p" in params["bcs"] and "Q_f" not in params:
    Delta_p = fem.Constant(mesh_, params["bcs"]["Delta p"])
    Delta_p_num = nums(Delta_p)[0]
    bc_right_phi = fem.dirichletbc(get_phi_bc(mesh_, sigma_l_num - Delta_p_num,
                                              E.f, phi_f0_num, nu_num, _left=False),
                                   right_dofs_list[0], V[0])
    bcs.append(bc_right_phi)
    bc_right_sigma = fem.dirichletbc(sigma_l_num - Delta_p_num, right_dofs_list[3], V[3])
    bcs.append(bc_right_sigma)
    bc_left_pf = fem.dirichletbc(Delta_p, left_dofs_list[5], V[5])
    # bcs.append(bc_left_pf)
else:
    print("Need exactly one of Q_f and Delta p prescribed, exiting...")
    sys.exit()

if "c_left" in params["bcs"]:
    bc_left_c = fem.dirichletbc(params["bcs"]["c_left"], left_dofs_list[2], V[2])
    bcs.append(bc_left_c)
if "c_right" in params["bcs"]:
    bc_right_c = fem.dirichletbc(params["bcs"]["c_right"], right_dofs_list[2], V[2])
    bcs.append(bc_right_c)
if "sigma_left" in params["bcs"]:
    bc_left_sigma = fem.dirichletbc(sigma_l, left_dofs_list[3], V[3])
    bcs.append(bc_left_sigma)
if "u_s_right" in params["bcs"]:
    bc_right_us = fem.dirichletbc(params["bcs"]["u_s_right"], right_dofs_list[4], V[4])
    bcs.append(bc_right_us)
if "p_f_right" in params["bcs"]:
    bc_right_pf = fem.dirichletbc(params["bcs"]["p_f_right"], right_dofs_list[5], V[5])
    bcs.append(bc_right_pf)

param_file.close()


def get_vs_from_E_phi(_mesh, _phi_f, _E, _a_list, _phi_f0, _nu,
                      _t_v, _t_vs, _t_phi, _delta_t):
    _, Q_f_arr = fenics_to_numpy(_mesh, Q_f)
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, E_arr = fenics_to_numpy(_mesh, _E)
    k_e_arr = compute_k_e(phi_f_arr, _phi_f0)
    g_arr = compute_g(phi_f_arr, _phi_f0, _nu)
    dg_dphi_arr = compute_dg_dphi(phi_f_arr, _phi_f0, _nu)
    da_dt_val = (_a_list[-1] - _a_list[-2]) / _delta_t
    print(da_dt_val)
    _a = _a_list[-1]
    prod = E_arr * dg_dphi_arr * np.gradient(phi_f_arr, xi_arr) + g_arr * np.gradient(E_arr, xi_arr)
    _v_s = (Q_f_arr / _t_v + phi_f_arr * k_e_arr * prod
            / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs
    return _v_s


def get_vs_from_u_phi(_mesh, _phi_f, _u_s_new, _u_s_old, _phi_f0, _a_list,
                      _t_vs, _t_sc, _delta_t):
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, u_s_new_arr = fenics_to_numpy(_mesh, _u_s_new)
    _, u_s_old_arr = fenics_to_numpy(_mesh, _u_s_old)
    dus_dt_arr = (u_s_new_arr - u_s_old_arr) / _delta_t
    da_dt_val = (_a_list[-1] - _a_list[-2]) / _delta_t
    print(da_dt_val)
    return (1 / (1 - phi_f_arr) *
            ((1 - _phi_f0) * dus_dt_arr - (1 - xi_arr) * da_dt_val * (phi_f_arr - _phi_f0))
            * _t_vs / _t_sc)


def get_sigma_from_E_g(_mesh, _E, _g, _phi_f0):
    """Computes the Terzaghi stress as a function of the Young's modulus
    and the porosity.

    :param _mesh: The mesh.
    :param _E: The Young's modulus.
    :param _g: The effective stress (function of porosity).
    :param _phi_f0: The initial porosity.
    :return: The Terzaghi stress.
    """
    _, E_arr = fenics_to_numpy(_mesh, _E)
    _, g_arr = fenics_to_numpy(_mesh, _g)
    return E_arr * g_arr / (1 - _phi_f0)


# This quantity is just for plotting purposes
v_s_ = Quantity("$v_{s}$", "YlOrBr", 7, mesh_)
# We don't know the initial array v_s
v_s_.f = np.full(N_x + 1, np.nan)
# quantities = [phi_f, E, c, sigma, u_s, v_s_, p_f]
quantities = [phi_f, E, p_f, sigma, u_s, v_s_, c]
# quantities = [phi_f, E, c]
# quantities = [u_s, p_f]


"""
Set up figure for the overall plot
"""
nrows, ncols = 4, 2
# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
# fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, (10 * nrows)/3), sharex=True)
plt.subplots_adjust(wspace=1.0 * (ncols - 1))
axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
# axs_list = [axs[i] for i in range(nrows)]
Quantity.set_axs(quantities, axs_list[:-1])
if log_time:
    norm = mpl.colors.LogNorm(vmin=float(t.c(0.0)), vmax=t_final)
else:
    norm = mpl.colors.Normalize(vmin=0.0, vmax=t_final)

"""
Plot the initial curves and save all our data
"""
# saving = [True, True, True, True]
# short_quants = ["phi", "E", "c", "u_s"]
# saving = [True, True, True, True, True]

for quantity in quantities:
    quantity.initialise_dataframe(xi_arr)

Quantity.plot_quantities(quantities, norm, 0.0, saving)

# define the time derivatives
dphi_dt = (phi_f.u - phi_f.u_old) / delta_t.c
dE_dt = (E.u - E.u_old) / delta_t.c
dc_dt = (c.u - c.u_old) / delta_t.c
da_dt = (a - a_old) / delta_t.c
dus_dt = (u_s.u - u_s.u_old) / delta_t.c

# Effective permeability and effective stress
k = (1 - phi_f0) ** 2 * (phi_f.u ** 3) / pow(phi_f0, 3) / (1 - phi_f.u) ** 2
k_div_phi = (1 - phi_f0) ** 2 * (phi_f.u ** 2) / pow(phi_f0, 3) / (1 - phi_f.u) ** 2
g = (((1 - phi_f0) / (1 - phi_f.u) - 2 * nu - (1 - 2 * nu) * (1 - phi_f.u) / (1 - phi_f0))
     / (2 * (1 + nu) * (1 - 2 * nu)))
dg_dphi = ((1 - phi_f0) / (1 - phi_f.u) ** 2 + (1 - 2 * nu) / (1 - phi_f0)) / (2 * (1 + nu) * (1 - 2 * nu))

# Controls the flux
alpha = 1.0
# vt = t_v * (Q_f / t_v_f + alpha * da_dt / t_sc)
# dEg_dx = g * E.u.dx(0) + E.u * dg_dphi * phi_f.u.dx(0)
dEg_dx = (E.u * g).dx(0)


# Find intermediate expressions for the solid and fluid velocities
# Below are two different expressions that we need for the solid velocity (they
# are equivalent definitions)
v_s = t_v_s * (v.u / t_v + k * p_f.u.dx(0) / ((1 - a) * t_phi))
_v_s = t_v_s / (1 - phi_f.u) * ((1 - phi_f0) * dus_dt / t_sc - (1 - xi) * da_dt * (phi_f.u - phi_f0) / t_sc)
v_f = t_v_f * (v.u / t_v - (1 - phi_f.u) * k_div_phi * dEg_dx / ((1 - a) * t_phi))
_v_f = t_v_f * (_v_s / t_v_s - k_div_phi * dEg_dx / ((1 - a) * t_phi))
_v = t_v * (_v_s / t_v_s - k * dEg_dx / ((1 - a) * t_phi))
phi_f_v_f = t_v_f * (_v - (1 - phi_f.u) * _v_s)
__v = t_v * (phi_f_v_f + (1 - phi_f.u) * _v_s)
# _v = Q_f

"""
Define the weak form
"""

# Weak form for the phi equation
Fun_phi = ((dphi_dt - da_dt * phi_f.u / (1 - a)) * phi_f.v / t_sc * dx +
           ((1 / (1 - a))**2 * (1 - phi_f.u) * k * dEg_dx / t_phi -
            (1 / (1 - a)) * phi_f.u * (_v / t_v - (1 - xi) * da_dt / t_sc)) * phi_f.v.dx(0) * dx +
           (1 / (1 - a)) * (_v / t_v) * phi_f.v * ds(2) -
           (1 / (1 - a)) * (_v / t_v - da_dt / t_sc) * phi_f.v * ds(1))
# Fun_phi = ((dphi_dt - da_dt * phi_f.u / (1 - a)) * phi_f.v / t_sc * dx +
#            ((1 / (1 - a))**2 * phi_f.u * k_e.u * dEg_dx / t_phi -
#             (1 / (1 - a)) * phi_f.u * (Q_f / t_v_f + xi * da_dt / t_sc)) * phi_f.v.dx(0) * dx +
#            (Q_f / t_v_f + da_dt / t_sc) * phi_f.v / (1 - a) * ds(2))
#            (Q_f / t_v_f) * phi_f.v / (1 - a) * ds(1))
# Fun_phi = ((dphi_dt - da_dt * (1 - xi) / (1 - a) * phi_f.u.dx(0)) / t_sc * phi_f.v * dx +
#            (phi_f.u * _v_s).dx(0) / (1 - a) / t_v_s * phi_f.v * dx +
#            phi_f.u * k_e * p_f.u.dx(0) * (1 - phi_f0) / ((1 - a) ** 2 * (1 - phi_f.u) * t_phi) * phi_f.v.dx(0) * dx)

# Weak form for the E equation
Fun_E = (dE_dt / t_sc + c.u * (E.u - E_min) / t_E
         + (_v_s / t_v_s - (1 - xi) * da_dt / t_sc) / (1 - a) * E.u.dx(0)) * E.v * dx
# Fun_E = dE_dt * E.v / t_sc * dx + c.u * (E.u - E_min) * E.v / t_E * dx

# Weak form for the c equation
# Fun_c = ((phi_f.u * dc_dt + dphi_dt * c.u -
#           da_dt * c.u * phi_f.u / (1 - a)) / t_sc * c.v * dx +
#          phi_f.u / (1 - a) *
#          (c.u.dx(0) / ((1 - a) * t_c) -
#           (_v_f / t_v_f - (1 - xi) * da_dt / t_sc) * c.u) * c.v.dx(0) * dx +
#          c.u * phi_f.u * _v_f / (1 - a) / t_v_f * c.v * ds(2))
# Weak form for the c equation with new boundary conditions
Fun_c = (((phi_f.u * dc_dt + dphi_dt * c.u -
          da_dt * c.u * phi_f.u / (1 - a)) / t_sc * c.v +
         1 / (1 - a) *
         (phi_f.u * c.u.dx(0) / ((1 - a) * t_c) -
          (phi_f.u * _v_f / t_v_f - phi_f.u * (1 - xi) * da_dt / t_sc) * c.u) * c.v.dx(0)) * dx +
         c.u * _v / (1 - a) / t_v_f * c.v * ds(2) +
         (c.u * da_dt / t_sc - _v * c_minus / t_v) / (1 - a) * c.v * ds(1))

# Weak form for the Terzaghi stress (sort of Lagrange multiplier)
Fun_sigma = (sigma.u - (E.u * g)) * sigma.v * dx

# Weak form for the displacement
Fun_us = ((u_s.u.dx(0) * u_s.v -
           (phi_f.u - phi_f0) * (1 - a) / (1 - phi_f0) * u_s.v) * dx)

# Weak form for the fluid pressure
Fun_pf = (sigma.u - p_f.u) * p_f.v.dx(0) * dx + (sigma.u - p_f.u) * p_f.v * ds(1)

# Weak form for the phase-averaged velocity
if "Q_f" in params:
    Fun_v = (((Q_f.c - v.u) / t_v) * v_v * dx)
else:
    Fun_v = (v.u - _v) * v_v * dx

# Weak form for the moving boundary
Fun_a = ((phi_f.u - 1) + (1 - phi_f0) / (1 - a)) * v_a * dx

# Combining the weak forms
Funs = [Fun_phi, Fun_E, Fun_c, Fun_sigma, Fun_us, Fun_pf, Fun_v, Fun_a]

"""
Loop over time steps and solve
"""
t_list = [float(t.c(0.0))]
# Lists of averages to record (Q_f, E_avg, c_avg, phi_f_avg)
Q_f_list = []
avgs_dict = {"phi_f": [phi_f.get_average()], "E": [E.get_average()], "c": [c.get_average()]}
c_right_list = [float(c_plus(0.0))]
# integral_v_list = [float(integral_v(0.0))]
t_fl = t_list[0]
for n in range(N_time):
    t_fl = float(t_next.c(0.0))
    problem = NonlinearProblem(Funs, w_list, bcs)
    solver = PETSc.NonlinearSolver().create(MPI.COMM_WORLD)
    solver.setType("newtonls")
    solver.setTolerances(rtol=1e-8)

    print("Time:", np.round(t_fl, 3))

    # Solve (iterate until BC for E is consistent)
    solver.solve(w.vector)
    phi_f.f, E.f, c.f, u_s_new, sigma.f, p_f.f, v.f, a_f = f_copy(w_list)
    _, phi_f_arr = fenics_to_numpy(mesh_, phi_f.f)
    a_list.append(a_f(0.0))

    # if n == 0:
    #     # We need an idea of what Q_f was at t = 0. We approximate this by whatever
    #     # Q_f is at t = delta_t.
    #     Q_f_list.append(float(fenics_to_numpy(mesh, v.f)[1][0]))
    Q_f_list.append(float(fenics_to_numpy(mesh_, v.f)[1][0]))
    c_right_list.append(float(fenics_to_numpy(mesh_, c.f)[1][-1]))

    # Update some variables
    t.update_val({"tau": (n + 1) * delta_tau})
    t_next.update_val({"tau": (n + 2) * delta_tau})
    t_list.append(float(t.c(0.0)))
    delta_t_fl = t_list[-1] - t_list[-2]

    # Record various outputs
    avgs_dict["phi_f"].append(phi_f.get_average())
    avgs_dict["E"].append(E.get_average())
    avgs_dict["c"].append(c.get_average())

    # sigma.f = get_sigma_from_E_g(mesh, E.f, compute_g(phi_f_arr, phi_f0_num, nu_num), phi_f0_num)
    # v_s_.f = get_vs_from_E_phi(mesh, phi_f.f, E.f, a_list, phi_f0_num, nu_num,
    #                            t_v_num, t_v_s_num, t_phi_num, delta_t_fl)
    v_s_.f = get_vs_from_u_phi(mesh_, phi_f.f, u_s_new, u_s.f, phi_f0_num, a_list,
                               t_v_s_num, t_sc_num, delta_t_fl)
    w_old.x.array[:] = w.x.array[:]

    # Change coordinates onto the fixed domain for plotting
    (phi_f.f_fixed, E.f_fixed, c.f_fixed, sigma.f_fixed,
     u_s.f_fixed, p_f.f_fixed, v_s_.f_fixed) = xi_t_to_x_t(mesh_, a_list[-1], phi_f, E,
                                                           c, sigma, u_s, p_f, v_s_)

    # plot at the current timepoint if needed
    if phi_f.f_fixed[-1] < 0.0:
        phi_f.f_fixed[-1] = 0.0
    phi_r = phi_f.f_fixed[-1]
    # phi_r_list.append(phi_r)
    if (n + 1) % plotting_freq == 0:
        Quantity.plot_quantities(quantities, norm, t_fl, saving,
                                 fixed_domain=fixed_domain)
    u_s.f = u_s_new

    phi_l = fenics_to_numpy(mesh_, phi_f.f)[1][0]
    if phi_l < 0.0:
        print("Porosity on the left has reached zero, exiting...")
        break
    bc_left_phi = fem.dirichletbc(get_phi_bc(mesh_, sigma_l_num, E.f, phi_f0_num, nu_num),
                                  left_dofs_list[0], V[0])
    bcs[0] = bc_left_phi
    phi_r = get_phi_bc(mesh_, sigma_l_num - Delta_p_num, E.f, phi_f0_num, nu_num,
                       _left=False)
    bc_right_phi = fem.dirichletbc(phi_r, right_dofs_list[0], V[0])
    bcs[1] = bc_right_phi

    if phi_r == 0.0:
        print("Porosity on the right has reached zero, exiting...")
        break

# Save all relevant quantities
if any(saving) and not os.path.isdir(data_path):
    os.makedirs(data_path)
file_names = [f"{data_path}/_{q}_{plot_coord}.csv" for q in short_quants]
Quantity.write_to_csv(quantities, saving, file_names)

"""
Creating the plots
"""

# Set up the colorbars and label the plots
Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex)

# Check plot directory exists
plot_path = f"resources/{parent}/{trial}/{sub_trial}/plots"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
# Save figure
fig.savefig(f"{plot_path}/_time_traces_{plot_coord}.png", bbox_inches="tight")

# Create figure for the imposed velocity and left boundary over time
fig_Q_a, axs_Q_a = plt.subplots(nrows=2, ncols=1, figsize=(8, 20 / 3), sharex=True)
ax_Q, ax_a = axs_Q_a
times = np.array(t_list)
ax_Q.plot(times[1:], np.array(Q_f_list), lw=2,
          color='forestgreen')
ax_Q.set_ylabel("$v(t)$")
ax_Q.set_xscale("log")
ax_Q.set_yscale("log")
ax_a.plot(times, np.array(a_list), lw=2,
          color='darkgoldenrod')
ax_a.set_ylabel("$a(t)$")
ax_a.set_xlabel("$t$")
fig_Q_a.savefig(f"{plot_path}/Q_a.png", bbox_inches="tight")

fig_tau, ax_tau = plt.subplots(nrows=1, ncols=1, figsize=(8, 10 / 3))
ax_tau.plot(np.linspace(0, N_time * delta_tau, len(times)), times, lw=2,
            color='darkviolet')
ax_tau.set_ylabel("$t$")
ax_tau.set_xlabel("$\\tau$")
ax_tau.set_yscale("log")
fig_tau.savefig(f"{plot_path}/tau.png", bbox_inches="tight")

# Create figure for various averages over time
fig_avgs, ax_all = plt.subplots(figsize=(8, 3))
phi_f_avg_arr, E_avg_arr, c_avg_arr = np.array(avgs_dict["phi_f"]), np.array(avgs_dict["E"]), np.array(avgs_dict["c"])

# Plot E_avg, c_avg and phi_r over time on the same axis
ax_all.plot(times, phi_f_avg_arr, lw=2,
            color="mediumblue", label="$\\overline{\\phi_{f}}$")
ax_all.plot(times, E_avg_arr, lw=2,
                color="indigo", label="$\\overline{E}$")
ax_all.plot(times, c_avg_arr, lw=2,
                color="red", label="$\\overline{c}$")
ax_all.set_xlabel("$t$")
ax_all.legend()
plt.subplots_adjust(hspace=0.3, wspace=0.4)
fig_avgs.savefig(f"{plot_path}/averages.png", bbox_inches="tight")
