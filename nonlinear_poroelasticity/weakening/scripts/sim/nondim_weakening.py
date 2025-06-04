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
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from nonlinear_poroelasticity.weakening.scripts import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Reading in our parameters
"""
trial = "long_steady_state"
sub_trial = "v_0_1"
param_file = open(f"resources/{trial}/{sub_trial}/params.json")
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

# The frequency of plotting
num_lines = N_time / 1
# num_lines = 50
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


# Setting up the moving boundary
a_list = [params["ics"]["a"]]


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

# For vars phi_f, E, c, sigma, u_s, p_f, v, a
element = MixedElement([P1, P1, P1, P1, P1, P1, P1, P0])
V = FunctionSpace(mesh, element)

# Time expression and time array for a simulation with changing timestep
t_tau = params["comp"]["t(tau)"] if "t(tau)" in params["comp"] else "tau"
t = Expression(t_tau, degree=1, tau=0.0, delta_tau=delta_tau, N_time=N_time, domain=mesh)
t_next = Expression(t_tau, degree=1, tau=delta_tau, delta_tau=delta_tau, N_time=N_time, domain=mesh)
delta_t = t_next - t
t_final = float(Expression(t_tau, degree=1, tau=N_time * delta_tau,
                           delta_tau=delta_tau, N_time=N_time, domain=mesh)(0.0))

# Imposed fluid flux
Q_f = Expression(params["v"]["expr"],
                 degree=1, t=t, delta_tau=delta_tau, N_time=N_time, domain=mesh)
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
sigma = Quantity("$\\sigma_{xx}'$", "Greys", 3, mesh)
u_s = Quantity("$u_s$", "Greens", 4, mesh)
p_f = Quantity("$p_f$", "Oranges", 5, mesh)
v = Quantity("$v$", "RdPu", 6, mesh)

# Set up the functions from the joint space
v_phi, v_E, v_c, v_sigma, v_us, v_pf, v_v, v_a = TestFunctions(V)


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


short_quants = ["phi", "E", "c", "sigma", "u_s", "p_f", "v", "v_s"]
# short_quants = ["phi", "E", "sigma", "u_s", "p_f", "v_s"]
data_path = f"resources/{trial}/{sub_trial}/data"
# ic_file_names = [f"{data_path}/{q}_xi.csv" for q in short_quants]
# ics = [pd.read_csv(ic_file_names[i]).to_numpy()[:, -1]
#        for i in range(len(short_quants))]

# Define the initial conditions
phi_f0_ic = 'phi_f0 - gamma * (1 - phi_f0) * (1 + nu) * (1 - 2 * nu) / (1 - nu) * x[0]'
E_ic = f'{params["ics"]["E"]} * (1 + 0 * x[0])'
w_0 = Expression(('phi_f0', E_ic, params["ics"]["c"],
                  '0.0', params["ics"]["u_s"], '0.0', '0.0', 'a_0'),
                 degree=1, phi_f0=phi_f0, a_0=a_list[0],
                 E_min=E_min, gamma=t_phi_num/t_v_num, nu=nu)
w_old = project(w_0, V)

# w_old = Function(V)
w = Function(V)
w_phi, w_E, w_c, w_sigma, w_us, w_pf, w_v, a = split(w)
phi_old, E_old, c_old, sigma_old, u_s_old, p_f_old, v_old, a_old = split(w_old)
# _phi_old, _E_old, _c_old, _sigma_old, _u_s_old, _a_old = w_old.split()
# w_old_list = [_phi_old, _E_old, _c_old, _sigma_old, _u_s_old, _a_old]
#
# # Set the initial conditions
# for i, f_old in enumerate(w_old_list):
#     f_old.interpolate(lambda y: retrieve_ic(y, ics[i]))
# _a_old.interpolate(lambda y: retrieve_ic(y, a_list))

phi_f.f, E.f, c.f, sigma.f, u_s.f, p_f.f, v.f, a_f = w_old.split(deepcopy=True)
phi_f.set_sym_functions(w_phi, v_phi, phi_old)
E.set_sym_functions(w_E, v_E, E_old)
sigma.set_sym_functions(w_sigma, v_sigma, sigma_old)
c.set_sym_functions(w_c, v_c, c_old)
u_s.set_sym_functions(w_us, v_us, u_s_old)
p_f.set_sym_functions(w_pf, v_pf, p_f_old)
v.set_sym_functions(w_v, v_v, v_old)

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
    b = 2 * (1 + _nu) * (1 - 2 * _nu) * _sigma_l / E_l + 2 * _nu
    discriminant = b ** 2 + 4 * (1 - 2 * _nu)
    factor = (1 - _phi_f0) / (2 * (1 - 2 * _nu))
    return 1 - factor * (discriminant ** (1 / 2) - b)


boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)


# Define a function for the left boundary; this function
# just needs to return the value true when xi is close to
# the boundary 0
class Left(SubDomain):
    def inside(self, xi, on_boundary):
        return near(xi[0], 0) and on_boundary


class Right(SubDomain):
    def inside(self, xi, on_boundary):
        return near(xi[0], 1) and on_boundary


# Define the boundary conditions at the left and right
left = Left()
left.mark(boundary_markers, 1)
right = Right()
right.mark(boundary_markers, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
bcs = []

sigma_l_num = params["bcs"]["sigma_left"]
sigma_l = Constant(sigma_l_num)
bc_left_phi = DirichletBC(V.sub(0), get_phi_l(mesh, sigma_l_num, E.f, phi_f0_num, nu_num), boundary_markers, 1)
bc_right_phi = DirichletBC(V.sub(0), get_phi_l(mesh, sigma_l_num, E.f, phi_f0_num, nu_num), boundary_markers, 2)

bcs.append(bc_left_phi)
if "c_left" in params["bcs"]:
    bc_left_c = DirichletBC(V.sub(2), params["bcs"]["c_left"], boundary_markers, 1)
    c.add_bc(bc_left_c)
    bcs.append(bc_left_c)
if "c_right" in params["bcs"]:
    bc_right_c = DirichletBC(V.sub(2), params["bcs"]["c_right"], boundary_markers, 2)
    c.add_bc(bc_right_c)
    bcs.append(bc_right_c)
if "sigma_left" in params["bcs"]:
    bc_left_sigma = DirichletBC(V.sub(3), params["bcs"]["sigma_left"], boundary_markers, 1)
    sigma.add_bc(bc_left_sigma)
    bcs.append(bc_left_sigma)
if "u_s_right" in params["bcs"]:
    bc_right_us = DirichletBC(V.sub(4), params["bcs"]["u_s_right"], boundary_markers, 2)
    u_s.add_bc(bc_right_us)
    bcs.append(bc_right_us)
if "p_f_right" in params["bcs"]:
    bc_right_pf = DirichletBC(V.sub(5), params["bcs"]["p_f_right"], boundary_markers, 2)
    p_f.add_bc(bc_right_pf)
    bcs.append(bc_right_pf)

bc_right_phi = DirichletBC(V.sub(0), 0.0, boundary_markers, 2)
# bc_left_phi = DirichletBC(V.sub(0), phi_f0_num, left)

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


def get_vs_from_E_phi(_mesh, _phi_f, _E, _a_list, _phi_f0, _nu,
                      _t_v, _t_vs, _t_phi, _alpha):
    _, Q_f_arr = fenics_to_numpy(_mesh, Q_f)
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, E_arr = fenics_to_numpy(_mesh, _E)
    k_e_arr = compute_k_e(phi_f_arr, _phi_f0)
    g_arr = compute_g(phi_f_arr, _phi_f0, _nu)
    dg_dphi_arr = compute_dg_dphi(phi_f_arr, _phi_f0, _nu)
    da_dt_val = (_a_list[-1] - _a_list[-2]) / delta_tau
    print(da_dt_val)
    _a = _a_list[-1]
    prod = E_arr * dg_dphi_arr * np.gradient(phi_f_arr, xi_arr) + g_arr * np.gradient(E_arr, xi_arr)
    _v_s = (Q_f_arr / _t_v + 0 * _alpha * da_dt_val / t_sc_num +
            phi_f_arr * k_e_arr * prod
            / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs
    return _v_s
    # return (phi_f_arr * k_e_arr * np.gradient(E_arr * g_arr, xi_arr)
    #         / ((1 - _a) * (1 - phi_f_arr) * _t_phi)) * _t_vs


def get_vs_from_u_phi(_mesh, _phi_f, _u_s_new, _u_s_old, _phi_f0, _a_list,
                      _t_vs, _t_sc):
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, u_s_new_arr = fenics_to_numpy(_mesh, _u_s_new)
    _, u_s_old_arr = fenics_to_numpy(_mesh, _u_s_old)
    dus_dt_arr = (u_s_new_arr - u_s_old_arr) / delta_tau
    da_dt_val = (_a_list[-1] - _a_list[-2]) / delta_tau
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


# E.f = Constant(params["ics"]["E"])
# sigma.f = Constant(0.0)
# This quantity is just for plotting purposes
v_s_ = Quantity("$v_{s}$", "YlOrBr", 7, mesh)
# We don't know the initial array v_s
v_s_.f = np.full(N_x + 1, np.nan)
# quantities = [phi_f, E, c, sigma, u_s, v_s_]
quantities = [phi_f, E, c, sigma, u_s, v_s_, p_f, v]
# quantities = [phi_f, v_s_, v]
# quantities = [u_s, p_f]


"""
Set up figure for the overall plot
"""
nrows, ncols = 4, 2
# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
# fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, (10 * nrows)/3), sharex=True)
plt.subplots_adjust(wspace=1.0 * (ncols - 1))
axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
# axs_list = [axs[i] for i in range(nrows)]
Quantity.set_axs(quantities, axs_list)
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
dphi_dt = (phi_f.u - phi_f.u_old) / delta_t
dE_dt = (E.u - E.u_old) / delta_t
# dsigma_dt = (sigma.u - sigma.u_old) / delta_t
dc_dt = (c.u - c.u_old) / delta_t
da_dt = (a - a_old) / delta_t
dus_dt = (u_s.u - u_s.u_old) / delta_t

# Effective permeability and effective stress
k_e = (1 - phi_f0) * (phi_f.u ** 2) / pow(phi_f0, 3) / (1 - phi_f.u)
g = (((1 - phi_f0) ** 2 / (1 - phi_f.u) - 2 * nu * (1 - phi_f0) - (1 - 2 * nu) * (1 - phi_f.u))
     / (2 * (1 + nu) * (1 - 2 * nu)))
dg_dphi = ((1 - phi_f0) ** 2 / (1 - phi_f.u) ** 2 + 1 - 2 * nu) / (2 * (1 + nu) * (1 - 2 * nu))
# E.u = (1 - phi_f0) * sigma.u / g.u

# dE_dt = (1 - phi_f0) * (dsigma_dt / g.u - sigma.u / g.u ** 2 * dg_dphi * dphi_dt)

# Controls the flux
alpha = 1.0
# vt = t_v * (Q_f / t_v_f + alpha * da_dt / t_sc)
dEg_dx = g * E.u.dx(0) + E.u * dg_dphi * phi_f.u.dx(0)

# Find intermediate expressions for the solid and fluid velocities
# Below are two different expressions that we need for the solid velocity (they
# are equivalent definitions)
v_s = t_v_s * (v.u / t_v +
               phi_f.u * k_e * p_f.u.dx(0) * (1 - phi_f0) / ((1 - a) * (1 - phi_f.u) * t_phi))
_v_s = t_v_s / (1 - phi_f.u) * ((1 - phi_f0) * dus_dt / t_sc - (1 - xi) * da_dt * (phi_f.u - phi_f0) / t_sc)
v_f = t_v_f * (v.u / t_v - k_e * p_f.u.dx(0) * (1 - phi_f0) / ((1 - a) * t_phi))

"""
Define the weak form
"""

# Weak form for the phi equation
Fun_phi = ((dphi_dt - da_dt * phi_f.u / (1 - a)) * phi_f.v / t_v * dx +
           ((1 / (1 - a))**2 * phi_f.u * k_e * (1 - phi_f0) * p_f.u.dx(0) / t_phi -
            (1 / (1 - a)) * phi_f.u * (v.u / t_v - (1 - xi) * da_dt / t_v)) * phi_f.v.dx(0) * dx +
           (1 / (1 - a)) * (v.u / t_v) * phi_f.v * ds(2) -
           (1 / (1 - a)) * (v.u / t_v - da_dt / t_sc) * phi_f.v * ds(1))
# Fun_phi = ((dphi_dt - da_dt * phi_f.u / (1 - a)) * phi_f.v / t_sc * dx +
#            ((1 / (1 - a))**2 * phi_f.u * k_e.u * dEg_dx / t_phi -
#             (1 / (1 - a)) * phi_f.u * (Q_f / t_v_f + xi * da_dt / t_sc)) * phi_f.v.dx(0) * dx +
#            (Q_f / t_v_f + da_dt / t_sc) * phi_f.v / (1 - a) * ds(2))
#            (Q_f / t_v_f) * phi_f.v / (1 - a) * ds(1))
# Fun_phi = ((dphi_dt - da_dt * (1 - xi) / (1 - a) * phi_f.u.dx(0)) / t_sc * phi_f.v * dx +
#            (phi_f.u * _v_s).dx(0) / (1 - a) / t_v_s * phi_f.v * dx +
#            v_s / (1 - a) / t_v_s * phi_f.v.dx(0) * dx -
#            da_dt / (1 - a) / t_sc * phi_f.v * ds(1))

# Weak form for the E equation
Fun_E = (dE_dt / t_sc + c.u * (E.u - E_min) / t_E
         + (v_s / t_v_s - (1 - xi) * da_dt / t_sc) / (1 - a) * E.u.dx(0)) * E.v * dx
# Fun_E = dE_dt * E.v / t_sc * dx + c.u * (E.u - E_min) * E.v / t_E * dx

# Weak form for the c equation
Fun_c = ((phi_f.u * dc_dt + dphi_dt * c.u -
          da_dt * c.u * phi_f.u / (1 - a)) / t_sc * c.v * dx +
         phi_f.u / (1 - a) *
         (c.u.dx(0) / ((1 - a) * t_c) -
          (v_f / t_v_f - (1 - xi) * da_dt / t_sc) * c.u) * c.v.dx(0) * dx)

# Weak form for the Terzaghi stress (sort of Lagrange multiplier)
Fun_sigma = (sigma.u - (E.u * g) / (1 - phi_f0)) * sigma.v * dx

# Weak form for the displacement
Fun_us = ((u_s.u.dx(0) * u_s.v -
           (phi_f.u - phi_f0) * (1 - a) / (1 - phi_f0) * u_s.v) * dx)

# Weak form for the fluid pressure
Fun_pf = (sigma.u - p_f.u) * p_f.v.dx(0) * dx + (sigma.u - p_f.u) * p_f.v * ds(1)

# Weak form for the phase-averaged velocity
Fun_v = (((Q_f - v.u) / t_v) * v.v * dx)

# Weak form for the moving boundary
Fun_a = ((phi_f.u - 1) * (1) + (1 - phi_f0) / (1 - a)) * v_a * dx
# Fun_a = (da_dt / t_sc + Q_f / t_v) * v_a * dx
# Fun_a = ((1 - xi) / t_sc * da_dt - 1 / t_v_s * v_s) * v_a * ds
# Fun_a = xi * (Q_f / t_v + da_dt / t_sc - phi_f.u * v_f / t_v) * v_a * ds

# Combining the weak forms
Fun = Fun_phi + Fun_E + Fun_c + Fun_sigma + Fun_us + Fun_pf + Fun_v + Fun_a


# Define the Jacobian, problem and solver
jacobian = derivative(Fun, w)

"""
Loop over time steps and solve
"""
t_list = [float(t(0.0))]
Q_f_list = [float(Q_f(0.0))]
t_fl = t_list[0]
for n in range(N_time):
    t_fl = float(t_next(0.0))
    problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
    solver = NonlinearVariationalSolver(problem)
    print("Time:", np.round(t_fl, 3))

    # Solve
    solver.solve()
    phi_f.f, E.f, c.f, sigma.f, u_s_new, p_f.f, v.f, a_f = w.split(deepcopy=True)
    _, phi_f_arr = fenics_to_numpy(mesh, phi_f.f)
    a_list.append(a_f(0.0))
    # sigma.f = get_sigma_from_E_g(mesh, E.f, compute_g(phi_f_arr, phi_f0_num, nu_num), phi_f0_num)
    v_s_.f = get_vs_from_E_phi(mesh, phi_f.f, E.f, a_list, phi_f0_num, nu_num,
                               t_v_num, t_v_s_num, t_phi_num, alpha)
    # v_s_.f = get_vs_from_u_phi(mesh, phi_f.f, u_s_new, u_s.f, phi_f0_num, a_list,
    #                            t_v_s_num, t_sc_num)
    w_old.assign(w)

    # Change coordinates onto the fixed domain for plotting
    (phi_f.f_fixed, E.f_fixed, c.f_fixed, sigma.f_fixed,
     u_s.f_fixed, p_f.f_fixed, v.f_fixed, v_s_.f_fixed) = xi_t_to_x_t(mesh, a_list[-1], phi_f, E,
                                                                      c, sigma, u_s, p_f, v, v_s_)

    # plot at the current timepoint if needed
    if phi_f.f_fixed[-1] < 0.0:
        phi_f.f_fixed[-1] = 0.0
    phi_r = phi_f.f_fixed[-1]
    if (n + 1) % plotting_freq == 0:
        Quantity.plot_quantities(quantities, norm, t_fl, saving,
                                 fixed_domain=fixed_domain)
    u_s.f = u_s_new

    # Update some variables
    t.tau += delta_tau
    t_next.tau += delta_tau
    t_list.append(float(t(0.0)))
    Q_f_list.append(float(Q_f(0.0)))

    phi_l = fenics_to_numpy(mesh, phi_f.f)[1][0]
    if phi_l < 0.0:
        print("Porosity on the left has reached zero, exiting...")
        break
    bc_left_phi = DirichletBC(V.sub(0), phi_l, boundary_markers, 1)
    bc_right_phi = DirichletBC(V.sub(0), phi_r, boundary_markers, 2)
    bcs[0] = bc_left_phi

    # if phi_r == 0.0:
    #     print("Porosity on the right has reached zero, exiting...")
    #     break

# Save all relevant quantities
if any(saving) and not os.path.isdir(data_path):
    os.makedirs(data_path)
file_names = [f"{data_path}/_{q}_{plot_coord}.csv" for q in short_quants]
Quantity.write_to_csv(quantities, saving, file_names)

# Set up the colorbars and label the plots
Quantity.annotate_plots(quantities, fig, norm, plot_coord_tex)

# Check plot directory exists
plot_path = f"resources/{trial}/{sub_trial}/plots"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Save figure
fig.savefig(f"{plot_path}/_time_traces_{plot_coord}.png", bbox_inches="tight")
# fig.savefig(f"{plot_path}/time_traces_constant_flux.png", bbox_inches="tight")

# Create figure for the imposed velocity and left boundary over time
fig_Q_a_tau, axs_Q_a_tau = plt.subplots(nrows=3, ncols=1, figsize=(8, 30 / 3), sharex=True)
ax_Q, ax_a, ax_tau = axs_Q_a_tau
times = np.array(t_list)
ax_Q.plot(times, np.array(Q_f_list), lw=2,
          color='forestgreen')
# ax_a.plot(np.array(v_s_0_list), times,
#           color='darkviolet', label='$v_s(0)$')
ax_Q.set_ylabel("$Q_f(t)$")
# a_expected = times * params["v"]["v_final"] * params["scales"]["t"] / t_v_num
ax_a.plot(times, np.array(a_list), lw=2,
          color='darkgoldenrod')
# ax_a.plot(np.array(v_s_0_list), times,
#           color='darkviolet', label='$v_s(0)$')
ax_a.set_ylabel("$a(t)$")
ax_a.set_xlabel("$t$")
ax_tau.plot(times, np.linspace(0, N_time * delta_tau, N_time + 1), lw=2,
            color='darkviolet')
ax_tau.set_ylabel("$\\tau$")
ax_tau.set_xlabel("$t$")
fig_Q_a_tau.savefig(f"{plot_path}/Q_a_tau.png", bbox_inches="tight")
# fig_v_a.savefig(f"{plot_path}/Q_f_and_a_constant_flux.png", bbox_inches="tight")
