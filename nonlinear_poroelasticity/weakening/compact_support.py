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

In this script, we consider the compact support solution, which occurs after some
time t = t_0 (when the porosity first becomes zero on the right boundary).
The initial conditions at t = t_0 are whatever they were at the end of the previous
simulation,

with boundary conditions (on a domain [a(t), b(t)] with left + right moving boundary):

        v_s = \\frac{t_{v_{s}}}{[t]}\\dot{a}(t) at x = a(t), v_s = 0 at x = b(t),
        phi_f = phi_l at x = a(t), phi_f = 0 at x = b(t),
        c = 1 at x = a(t), \\frac{1}{t_{c}}\\frac{\\p c}{\\p x} - \\frac{1}{t_{v_{f}}}cv_{f} = 0 at x = 1,

The left moving boundary can be determined by the following implicit relation:

        a(t) = \\phi_{f,0} - \\int_{a(t)}^{b(t)}\\phi_{f}(x, t)dx,

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
trial = "nondim_realistic_params"
sub_trial = "lower_bound"
param_file = open(f"resources/{trial}/{sub_trial}/params.json")
params = json.load(param_file)

# Whether we'll plot on a fixed domain or not
fixed_domain = False
plot_coord = "x" if fixed_domain else "xi"
plot_coord_tex = "$x$" if fixed_domain else "$\\xi$"
num_quants = 6

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
num_lines = N_time / 1
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

# Degradation parameter
beta_E = Constant(params["phys"]["beta_E"])

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
v_f_star = Constant(params["scales"]["v_f"])
v_s_star = Constant(params["scales"]["v_s"])

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star
t_v_f = L / v_f_star
t_v_s = L / v_s_star
t_E = 1 / (beta_E * c_star)
t_c = L ** 2 / D_m

x = Expression('x[0]', degree=1)


def nums(*constants: Constant):
    return tuple([constant(0.0) for constant in constants])


t_sc_num, phi_f0_num, nu_num = nums(t_sc, phi_f0, nu)
t_v_num, t_v_f_num, t_v_s_num = nums(t_v, t_v_f, t_v_s)
t_phi_num, t_E_num, t_c_num = nums(t_phi, t_E, t_c)

# Reading in from the previous simulation up to time t = t_0
short_quants = ["phi", "E", "c", "sigma", "u_s", "v_s"]
data_path = f"resources/{trial}/{sub_trial}/data"
file_names_xi = [f"{data_path}/{q}_xi.csv" for q in short_quants]
old_arrs = {short_quants[i]: pd.read_csv(file_names_xi[i]).to_numpy()[:, 2:]
            for i in range(len(short_quants))}
# Time at which compact support solution begins
t_0 = (old_arrs["phi"].shape[1] - 1) * delta_t
old_arrs_t_0 = {q: old_arrs[q][:, -1] for q in short_quants}
a_t_0 = old_arrs_t_0["u_s"][0]

# Setting up the moving boundaries
a_list = [a_t_0]
b_list = [1.0]

# Imposed phase-averaged velocity
vt = Expression(params["v"]["expr"],
                degree=1, t=t_0, delta_t=delta_t, N_time=N_time)

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

# For vars phi_f, E, c, sigma, u_s, a, b
element = MixedElement([P1, P1, P1, P1, P1, P0, P0])
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


def xi_t_to_x_t(_mesh: Mesh, _a: float, _b: float, *_quantities: Quantity):
    """Change the quantity from (xi, t) coordinates to (x, t) where
    \\xi = \\frac{x - a}{b - a}. We also fit onto the new mesh, which
    will involve some interpolation.

    :param _mesh: The mesh for the domain.
    :param _a: The left moving boundary a(t).
    :param _b: The right moving boundary b(t).
    :param _quantities: A tuple of quantities in (xi, t) coordinates (np array).
    :return: The new tuple of arrays in (x, t) coordinates.
    """
    f_part_list = []
    for quantity in _quantities:
        _, _f = fenics_to_numpy(_mesh, quantity.f)
        # How many points there are in the (x, t) domain
        N_part = N_x - int(_a * N_x) - int((1 - _b) * N_x)
        # Shrink region to [0, 1] (using transformation) and interpolate onto
        # only the left part of the grid
        f_part = np.interp(np.linspace(0, 1, N_part + 1),
                           np.linspace(0, 1, N_x + 1),
                           _f)
        # Fill the left of array with NaNs if domain has been compressed and
        # the right of array with zeroes for compact support
        if _a >= 0:
            f_part = np.concatenate([np.full(int(_a * N_x), np.nan), f_part,
                                     np.full(int((1 - _b) * N_x), 0.0)])
            quantity.mesh_fixed = np.linspace(0, 1, N_x + 1)
        # Else, if domain has expanded, we must change the mesh
        else:
            dx = 1 / N_x
            N_neg = - int(_a * N_x)
            mesh_fixed = np.linspace(- N_neg * dx, 1, N_x + N_neg + 1)
            # Update the fixed mesh of the quantity
            quantity.mesh_fixed = mesh_fixed
            # Add zeroes on the right for compact support
            f_part = np.concatenate([f_part, np.full(int((1 - _b) * N_x), 0.0)])

        f_part_list.append(f_part)

    return tuple(f_part_list)


"""
Define the solutions phi_f, E and c
"""

phi_f = Quantity("$\\phi_{f}$", "Blues", 0, mesh)
E = Quantity("$E$", "Purples", 1, mesh)
c = Quantity("$c$", "Reds", 2, mesh)
sigma = Quantity("$\\sigma_{xx}'$", "Greys", 3, mesh)
u_s = Quantity("$u_s$", "Greens", 4, mesh)

# Set up the functions from the joint space
v_phi, v_E, v_c, v_sigma, v_us, v_a, v_b = TestFunctions(V)

# Define the initial conditions
w_old = Function(V)
# !!! IMPORTANT !!! Numpy and FEniCS do not have the same way of indexing. We
# need to reverse the arrays back-to-front and insert them in an alternating
# fashion to convert between the two
reversed_initial_vals = np.array([old_arrs_t_0[q][::-1] for q in short_quants[:-1]])
right_cols = [np.array([reversed_initial_vals[0, 0]] + reversed_initial_vals[:, 1].tolist()
                       + reversed_initial_vals[1:, 0].tolist())]
initial_vals = np.concatenate(right_cols + [reversed_initial_vals[:, j] for j in range(2, N_x + 1)] +
                              [np.array(a_list)] + [np.array(b_list)])
print(initial_vals)
w_old.vector().set_local(initial_vals)
np.set_printoptions(threshold=np.inf)
print(w_old.vector()[:15])
for i, q in enumerate(short_quants[:-1]):
    print(old_arrs_t_0[q], len(old_arrs_t_0[q]))
    w_old.sub(i).vector().set_local(old_arrs_t_0[q])
    w_old.sub(i).vector().apply("insert")
    print(w_old.sub(i).vector()[:], len((w_old.sub(i).vector()[:])))
w_old.sub(5).vector().set_local(a_list)
w_old.sub(6).vector().set_local(b_list)
# w_0 = Expression(('phi_f0', params["ics"]["E"], params["ics"]["c"],
#                   '0.0', params["ics"]["u_s"], 'a_0', 'b_0'),
#                  degree=1, phi_f0=phi_f0, a_0=a_list[0], b_0=b_list[0], E_min=E_min)
# w_old = project(w_0, V)

# Set up all the functions
w = Function(V)
w_phi, w_E, w_c, w_sigma, w_us, a, b = split(w)
phi_old, E_old, c_old, sigma_old, u_s_old, a_old, b_old = split(w_old)
phi_f.f, E.f, c.f, sigma.f, u_s.f, a_f, b_f = w_old.split(deepcopy=True)
phi_f.set_sym_functions(w_phi, v_phi, phi_old)
E.set_sym_functions(w_E, v_E, E_old)
sigma.set_sym_functions(w_sigma, v_sigma, sigma_old)
c.set_sym_functions(w_c, v_c, c_old)
u_s.set_sym_functions(w_us, v_us, u_s_old)

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


k_e = Quantity("$k_{e}(\\phi_{f})$", "GnBu", 5, mesh)
g = Quantity("$\\sigma_{e}(\\phi_{f})$", "YlOrBr", 6, mesh)

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

sigma_l_num = params["bcs"]["sigma_left"]
sigma_l = Constant(sigma_l_num)
if "sigma_left" in params["bcs"]:
    bc_left_sigma = DirichletBC(V.sub(3), params["bcs"]["sigma_left"], left)
    sigma.add_bc(bc_left_sigma)
    bcs.append(bc_left_sigma)
if "c_left" in params["bcs"]:
    bc_left_c = DirichletBC(V.sub(2), params["bcs"]["c_left"], left)
    c.add_bc(bc_left_c)
    bcs.append(bc_left_c)
if "u_s_right" in params["bcs"]:
    bc_right_us = DirichletBC(V.sub(4), params["bcs"]["u_s_right"], right)
    u_s.add_bc(bc_right_us)
    bcs.append(bc_right_us)


def get_phi_l(_mesh, _sigma_l, _E, _phi_f0, _nu):
    """Finds the value of phi_f on the left.

    :param _mesh: The mesh.
    :param _sigma_l: The value of sigma' on the left.
    :param _E: The full Young's modulus profile.
    :param _phi_f0: The initial porosity.
    :param _nu: The Poisson's ratio.
    :return: The value of the porosity on the left.
    """
    _, E_arr = fenics_to_numpy(_mesh, _E)
    E_l = float(E_arr[0])
    # print(E_arr, "len E", len(E_arr))
    b = 2 * (1 + _nu) * (1 - 2 * _nu) * _sigma_l / E_l + 2 * _nu
    discriminant = b ** 2 + 4 * (1 - 2 * _nu)
    factor = (1 - _phi_f0) / (2 * (1 - 2 * _nu))
    _phi_l = 1 - factor * (discriminant ** (1 / 2) - b)
    return _phi_l


bc_left_phi = DirichletBC(V.sub(0), get_phi_l(mesh, sigma_l_num, E.f, phi_f0_num, nu_num), left)
bc_right_phi = DirichletBC(V.sub(0), 0.0, right)

bcs.append(bc_left_phi)
# bcs.append(bc_right_phi)

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
    g_arr = compute_g(phi_f_arr, _phi_f0, _nu)
    return (vt_arr / _t_v +
            phi_f_arr * k_e_arr * np.gradient(E_arr * g_arr, xi_arr)
            / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs
    # return (phi_f_arr * k_e_arr * np.gradient(E_arr * g_arr, xi_arr)
    #         / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs


def get_vs_from_u_phi(_mesh, _phi_f, _u_s_new, _u_s_old, _phi_f0, _a_list,
                      _b_list, _t_vs, _t_sc):
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, u_s_new_arr = fenics_to_numpy(_mesh, _u_s_new)
    _, u_s_old_arr = fenics_to_numpy(_mesh, _u_s_old)
    dus_dt_arr = (u_s_new_arr - u_s_old_arr) / delta_t
    da_dt_val = (_a_list[-1] - _a_list[-2]) / delta_t
    db_dt_val = (_b_list[-1] - _b_list[-2]) / delta_t
    return (1 / (1 - phi_f_arr) *
            ((1 - _phi_f0) * dus_dt_arr - ((1 - xi_arr) * da_dt_val + xi_arr * db_dt_val) * (phi_f_arr - _phi_f0))
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


# E.f = Constant(params["ics"]["E"])
# sigma.f = Constant(0.0)
v_s_ = Quantity("$v_{s}$", "YlOrBr", 7, mesh)
v_s_.f = old_arrs_t_0["v_s"]
quantities = [phi_f, E, c, sigma, u_s, v_s_]

for q in quantities:
    _, farr = fenics_to_numpy(mesh, q.f)
    print(q)
    print(farr[-3:])


"""
Set up figure for the overall plot
"""
nrows, ncols = 3, 2
# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
# fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
plt.subplots_adjust(wspace=1.0 * (ncols - 1))
axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
Quantity.set_axs(quantities, axs_list)
norm = mpl.colors.Normalize(vmin=t_0, vmax=N_time * delta_t)

times = np.linspace(t_0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves and save all our data
"""
# saving = [True, True, True, True]
# short_quants = ["phi", "E", "c", "u_s"]
# saving = [True, True, True, True, True]

for quantity in quantities:
    quantity.initialise_dataframe(xi_arr)

Quantity.plot_quantities(quantities, norm, t_0, saving)

# define the time derivatives
dphi_dt = (phi_f.g - phi_f.g_old) / delta_t
dE_dt = (E.g - E.g_old) / delta_t
# dsigma_dt = (sigma.g - sigma.g_old) / delta_t
dc_dt = (c.g - c.g_old) / delta_t
da_dt = (a - a_old) / delta_t
db_dt = (b - b_old) / delta_t
dus_dt = (u_s.g - u_s.g_old) / delta_t

k_e.g = compute_k_e(phi_f.g, phi_f0)
g.g = compute_g(phi_f.g, phi_f0, nu)
# dg_dphi = compute_dg_dphi(phi_f.g, phi_f0, nu)
# E.g = (1 - phi_f0) * sigma.g / g.g

# dE_dt = (1 - phi_f0) * (dsigma_dt / g.g - sigma.g / g.g ** 2 * dg_dphi * dphi_dt)

# Find intermediate expressions for the solid and fluid velocities
# v_s = t_v_s * (vt / t_v +
#                phi_f.g * k_e.g * (E.g * g.g).dx(0) / ((b - a) * (1 - phi_f.g) * t_phi))
# v_f = t_v_f * (vt / t_v - k_e.g * (E.g * g.g).dx(0) / ((b - a) * t_phi))
v_s = t_v_s * (vt / t_v +
               phi_f.g * k_e.g * (E.g * g.g).dx(0) / ((1 - a) * (1 - phi_f.g) * t_phi))
v_f = t_v_f * (vt / t_v - k_e.g * (E.g * g.g).dx(0) / ((1 - a) * t_phi))

"""
Define the weak form
"""

# # Weak form for the phi equation
# Fun_phi = ((dphi_dt + (db_dt - da_dt) * phi_f.g / (b - a)) * phi_f.v / t_sc * dx +
#            ((1 / (b - a))**2 * phi_f.g * k_e.g * (E.g * g.g).dx(0) / t_phi -
#             (1 / (b - a)) * phi_f.g * (vt / t_v - ((1 - xi) * da_dt + xi * db_dt) / t_sc)) * phi_f.v.dx(0) * dx +
#            (vt / t_v - ((1 - xi) * da_dt + xi * db_dt) / t_sc) * phi_f.v / (b - a) * ds)
#
# # Fun_sigma = (sigma_l - (E.g * g.g) / (1 - phi_f0)) * (1 - xi) * sigma.v * ds
#
# # Weak form for the E equation
# Fun_E = (dE_dt / t_sc + c.g * (E.g - E_min) / t_E
#          + (v_s / t_v_s - ((1 - xi) * da_dt + xi * db_dt) / t_sc) / (b - a) * E.g.dx(0)) * E.v * dx
# # Fun_E = dE_dt * E.v / t_sc * dx + c.g * (E.g - E_min) * E.v / t_E * dx
#
# # Weak form for the c equation
# Fun_c = ((phi_f.g * dc_dt + dphi_dt * c.g +
#           (db_dt - da_dt) * c.g * phi_f.g / (b - a)) / t_sc * c.v * dx +
#          phi_f.g / (b - a) *
#          (c.g.dx(0) / ((b - a) * t_c) -
#           (v_f / t_v_f - ((1 - xi) * da_dt + xi * db_dt) / t_sc) * c.g) * c.v.dx(0) * dx)
#
# # Weak form for the Terzaghi stress (sort of Lagrange multiplier)
# Fun_sigma = (sigma.g - (E.g * g.g) / (1 - phi_f0)) * sigma.v * dx
#
# # Weak form for the displacement
# Fun_us = ((u_s.g.dx(0) * u_s.v -
#            (phi_f.g - phi_f0) * (b - a) / (1 - phi_f0) * u_s.v) * dx)
#
# # Weak form for the left moving boundary (both forms below are valid)
# Fun_a = (phi_f.g - 1 + (b - phi_f0) / (b - a)) * v_a * dx
# # Fun_a = ((1 - xi) / t_sc * da_dt - 1 / t_v_s * v_s) * v_a * ds

# Weak form for the phi equation
Fun_phi = ((dphi_dt - da_dt * phi_f.g / (1 - a)) * phi_f.v / t_sc * dx +
           ((1 / (1 - a))**2 * phi_f.g * k_e.g * (E.g * g.g).dx(0) / t_phi -
            (1 / (1 - a)) * phi_f.g * (vt / t_v - (1 - xi) * da_dt / t_sc)) * phi_f.v.dx(0) * dx +
           (vt / t_v - (1 - xi) * da_dt / t_sc) * phi_f.v / (1 - a) * ds)

# Fun_sigma = (sigma_l - (E.g * g.g) / (1 - phi_f0)) * (1 - xi) * sigma.v * ds

# Weak form for the E equation
Fun_E = (dE_dt / t_sc + c.g * (E.g - E_min) / t_E
         + (v_s / t_v_s - (1 - xi) * da_dt / t_sc) / (1 - a) * E.g.dx(0)) * E.v * dx
# Fun_E = dE_dt * E.v / t_sc * dx + c.g * (E.g - E_min) * E.v / t_E * dx

# Weak form for the c equation
Fun_c = ((phi_f.g * dc_dt + dphi_dt * c.g -
          da_dt * c.g * phi_f.g / (1 - a)) / t_sc * c.v * dx +
         phi_f.g / (1 - a) *
         (c.g.dx(0) / ((1 - a) * t_c) -
          (v_f / t_v_f - (1 - xi) * da_dt / t_sc) * c.g) * c.v.dx(0) * dx)

# Weak form for the Terzaghi stress (sort of Lagrange multiplier)
Fun_sigma = (sigma.g - (E.g * g.g) / (1 - phi_f0)) * sigma.v * dx

# Weak form for the displacement
Fun_us = ((u_s.g.dx(0) * u_s.v -
           (phi_f.g - phi_f0) * (1 - a) / (1 - phi_f0) * u_s.v) * dx)

# Weak form for the moving boundary (both forms below are valid)
Fun_a = (phi_f.g - 1 + (1 - phi_f0) / (1 - a)) * v_a * dx
# Fun_a = ((1 - xi) / t_sc * da_dt - 1 / t_v_s * v_s) * v_a * ds

# Weak form for the right moving boundary
# Fun_b = (xi * (b - a - t_v * phi_f.g * k_e.g * (E.g * g.g).dx(0) / (vt * (1 - phi_f.g) * t_phi))) * v_b * ds
Fun_b = (b - b_list[0]) * v_b * dx

# Combining the weak forms
Fun = Fun_phi + Fun_E + Fun_c + Fun_us + Fun_a + Fun_b + Fun_sigma


# Define the Jacobian, problem and solver
jacobian = derivative(Fun, w)

"""
Loop over time steps and solve
"""
v_list = [float(vt(0.0))]
for n in range(int(t_0 / delta_t), N_time):
    problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
    solver = NonlinearVariationalSolver(problem)
    print("Time:", np.round(n * delta_t, 3))

    # Update some variables
    vt.t = n * delta_t

    # Solve
    solver.solve()
    phi_f.f, E.f, c.f, sigma.f, u_s_new, a_f, b_f = w.split(deepcopy=True)
    _, phi_f_arr = fenics_to_numpy(mesh, phi_f.f)
    # sigma.f = get_sigma_from_E_g(mesh, E.f, compute_g(phi_f_arr, phi_f0_num, nu_num), phi_f0_num)
    # v_s_.f = get_vs_from_E_phi(mesh, phi_f.f, E.f, a_f(0.0), phi_f0_num, nu_num,
    #                            t_v_num, t_v_s_num, t_phi_num)
    a_list.append(a_f(0.0))
    b_list.append(b_f(0.0))
    v_s_.f = get_vs_from_u_phi(mesh, phi_f.f, u_s_new, u_s.f, phi_f0_num, a_list, b_list,
                               t_v_s_num, t_sc_num)
    v_list.append(float(vt(0.0)))
    w_old.assign(w)

    # Change coordinates onto the fixed domain for plotting
    (phi_f.f_fixed, E.f_fixed, c.f_fixed, sigma.f_fixed,
     u_s.f_fixed, v_s_.f_fixed) = xi_t_to_x_t(mesh, a_list[-1], b_list[-1], phi_f, E,
                                              c, sigma, u_s, v_s_)
    # plot at the current timepoint if needed
    if phi_f.f_fixed[-1] < 0.0:
        phi_f.f_fixed[-1] = 0.0
    phi_r = phi_f.f_fixed[-1]
    if (n + 1) % plotting_freq == 0:
        Quantity.plot_quantities(quantities, norm, (n + 1) * delta_t, saving,
                                 fixed_domain=fixed_domain)
    u_s.f = u_s_new
    phi_l = get_phi_l(mesh, sigma_l_num, E.f, phi_f0_num, nu_num)
    if phi_l < 0.0:
        print("Porosity on the left has reached zero, exiting...")
        break
    bc_left_phi = DirichletBC(V.sub(0), phi_l, left)
    bcs[-1] = bc_left_phi
    if phi_r < 0.0:
        print("Porosity on the right has reached zero, exiting...")
        break

# Save all relevant quantities
file_names = [f"{data_path}/{q}_{plot_coord}.csv" for q in short_quants]
# Quantity.write_to_csv(quantities, saving, file_names)

# Set up the colorbars and label the plots
Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex)

# Check plot directory exists
plot_path = f"resources/{trial}/{sub_trial}/plots"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Save figure
fig.savefig(f"{plot_path}/time_traces_{plot_coord}_cs.png", bbox_inches="tight")

# Create figure for the imposed velocity and left boundary over time
fig_v_a_b, axs_v_a_b = plt.subplots(nrows=3, ncols=1, figsize=(8, 30/3), sharex=True)
ax_v, ax_a, ax_b = axs_v_a_b
times = np.linspace(t_0, len(a_list) * delta_t + t_0, len(a_list))
ax_v.plot(times, np.array(v_list),
          color='forestgreen', label='$v(t)$')
# ax_a.plot(np.array(v_s_0_list), times,
#           color='darkviolet', label='$v_s(0)$')
ax_v.set_ylabel("Imposed velocity")
ax_v.legend()
# a_expected = times * params["v"]["v_final"] * params["scales"]["t"] / t_v_num
ax_a.plot(times, np.array(a_list),
          color='darkviolet', label='$a(t)$')
# ax_a.plot(np.array(v_s_0_list), times,
#           color='darkviolet', label='$v_s(0)$')
ax_a.set_xlabel("Time")
ax_a.set_ylabel("Left boundary")
ax_a.legend()
# ax_b.set_xlim(min(b_list), max(b_list))
ax_b.plot(times, np.array(b_list),
          color='darkgoldenrod', label='$b(t)$')
# ax_b.plot(np.array(v_s_0_list), times,
#           color='darkgoldenrod', label='$v_s(0)$')
ax_b.set_xlabel("Time")
ax_b.set_ylabel("Right boundary")
ax_b.legend()
# ax_b.set_xlim(min(b_list), max(b_list))
fig_v_a_b.savefig(f"{plot_path}/v_a_and_b.png", bbox_inches="tight")
