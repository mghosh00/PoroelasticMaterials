"""
This Python code solves the following nonlinear general system for the porosity,
Young's modulus and solute concentration

        \\frac{D^{f}\\phi_{f}}{Dt} = \\phi_{f}\\frac{\\p}{\\p x}\\left[
        k_{e}(\\phi_{f})\\frac{\\p}{\\p x}\\left(E\\sigma_{e}(\\phi_{f})\\right)\\right],
        \\frac{D^{s}E}{Dt} = -\\beta_{E}cE,
        \\phi_{f}\\frac{D^{f}c}{Dt} = \\frac{\\p}{\\p x}\\left(
        \\mathcal{D}_{m}\\phi_{f}\\frac{\\p c}{\\p x}\\right),

where the operators D^{f} and D^{s} are the material derivatives for the fluid and solid
fractions respectively. These operators are dependent on E, k_{e}, \\sigma_{e} (both given
functions of the porosity) and v(t), which is a prescribed phase-averaged velocity.

The initial conditions are at t = 0:

        \\phi_{f} = \\phi_{f,0} (= const.), E = E_{0}(x), c = c_{0}(x),

with boundary conditions (on a domain [a(t), L] with left moving boundary):

        v_s = \\dot{a}(t) at x = a(t), v_s = 0 at x = L,
        c = c^{*} at x = a(t), \\mathcal{D}_{m}\\frac{\\p c}{\\p x} - cv_{f} = 0 at x = a(t),

The moving boundary can be determined by the following implicit relation:

        a(t) = \\phi_{f,0}L - \\int_{a(t)}^{L}\\phi_{f}(x, t)dx,

given a known profile for \\phi_{f} at the previous timestep (in the numerical scheme).
We will also change coordinates onto a fixed domain (see details below).
"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

"""
Define model parameters
"""

# Length of domain, L
L = Constant(1)

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(0.5)

# Degradation parameter
beta_E = Constant(1)

# Diffusive parameter for the solute concentration
D_m = Constant(0.5)

# Poisson ratio and viscosity
nu = Constant(0.3)
mu = Constant(1)

# Permeability scale
k_0 = Constant(1)

# Fixed concentration on the left
c_star = Constant(1.0)

x = Expression('x[0]', degree=1)


def nums(*constants: Constant):
    return tuple([float(constant.values()[0]) for constant in constants])


L_num, nu_num, mu_num, k0_num = nums(L, nu, mu, k_0)
phi_f0_num, beta_E_num, D_m_num = nums(phi_f0, beta_E, D_m)

# Setting up the moving boundary
a_list = [0.0]

"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 100

# Number of mesh points
N_x = 100

# Imposed phase-averaged velocity
vt_0 = '0.0'
vt_small = '1e-2'
vt_step = 't < delta_t * N_time / 2 ? 0.0 : 0.1'
vt_cts_small = '0.01 * t'
vt = Expression(vt_small,
                degree=1, t=0.0, delta_t=delta_t, N_time=N_time)

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the xi coodinates
xi = SpatialCoordinate(mesh)[0]
xi_arr = np.linspace(0, 1, 101)

# Set up function space
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
P0 = FiniteElement("R", mesh.ufl_cell(), 0)

# For vars phi_f, E, c, u_s, a
element = MixedElement([P1, P1, P1, P1, P0])
V = FunctionSpace(mesh, element)

"""
Function to change spatial coordinates and get to the correct mesh.
"""


def xi_t_to_x_t(f: np.array, a: np.array):
    """Change the quantity from (xi, t) coordinates to (x, t) where
    \\xi = 1 - \\frac{L - x}{L - a(t)}. We also fit onto the new mesh, which
    will involve some interpolation.

    :param f: Some quantity in (xi, t) coordinates (np array).
    :param a: The moving boundary a(t).
    :return: The new array in (x, t) coordinates.
    """
    f_new = []

    # Now focus on each timestep
    for i in range(f.shape[1]):
        # Find the specific timestep
        f_i = f[:, i]
        # How many points there are in the (x, t) domain
        N_part = N_x - int(a[i] * N_x)
        # Shrink region to [0, 1] (using transformation) and interpolate onto
        # only the left part of the grid
        f_i_part = np.interp(np.linspace(0, 1, N_part + 1),
                             np.linspace(0, 1, N_x + 1),
                             f_i)
        # Fill the left of array with NaNs
        f_i_part = np.concatenate([np.full(N_x - N_part, np.nan), f_i_part])

        f_new.append(f_i_part)
    return np.array(f_new).transpose()


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
w_0 = Expression(('phi_f0', '1', '1', '0', 'a_0'),
                 degree=1, phi_f0=phi_f0, a_0=a_list[0])
w_old = project(w_0, V)


w = Function(V)
w_phi, w_E, w_c, w_us, a = split(w)
phi_f.f, E.f, c.f, u_s.f, a_f = w_old.split(deepcopy=True)
phi_old, E_old, c_old, u_s_old, a_old = split(w_old)
phi_f.set_sym_functions(w_phi, v_phi, phi_old)
E.set_sym_functions(w_E, v_E, E_old)
c.set_sym_functions(w_c, v_c, c_old)
u_s.set_sym_functions(w_us, v_us, u_s_old)

# Define also the known functions k_{e} and \\sigma_{e} (of porosity)


def compute_k_e(_phi_f, _phi_f0, _k_0, _mu):
    """Computes k_e as a function of the porosity.

    :param _phi_f: The porosity.
    :param _phi_f0: The initial porosity.
    :param _k_0: The permeability scale.
    :param _mu: The viscosity.
    :return: The effective permeability.
    """
    numerator = _k_0 * (1 - _phi_f0) * (_phi_f ** 2)
    denominator = _mu * (_phi_f0 ** 3) * (1 - _phi_f)
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
bc_left_c = DirichletBC(V.sub(2), c_star, left)
c.add_bc(bc_left_c)
# bcs = [bc_left_c]
# bcs = []

bc_right_us = DirichletBC(V.sub(3), 0, right)
u_s.add_bc(bc_right_us)
bcs = [bc_left_c, bc_right_us]


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


# Define our fluid and solid velocities on the right
# qt = 0
# _, v_s0_array = get_vs_R(mesh, phi_f1_R.f, x_c[0], phi_f0, qt, D_phi)
# _, v_f0_array = get_vf_R(mesh, v_s0_array, phi_f0, qt)
# v_s0_R = Quantity("$v_{s,0}^{R}$", "Oranges", 6, mesh)
# v_f0_R = Quantity("$v_{f,0}^{R}$", "PuRd", 7, mesh)
# v_s0_R.f = v_s0_array
# v_f0_R.f = v_f0_array


def get_vs_from_E_phi(_mesh, _phi_f, _E, _a, _phi_f0, _k0, _mu, _nu, _L):
    _, vt_arr = fenics_to_numpy(_mesh, vt)
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, E_arr = fenics_to_numpy(_mesh, _E)
    k_e_arr = compute_k_e(phi_f_arr, _phi_f0, _k0, _mu)
    sigma_e_arr = compute_sigma_e(phi_f_arr, _phi_f0, _nu)
    return (vt_arr +
            phi_f_arr * k_e_arr * np.gradient(E_arr * sigma_e_arr, xi_arr)
            / ((_L - _a) * (1 - phi_f_arr)))


def get_vs_from_u_phi(_mesh, _phi_f, _u_s_new, _u_s_old, _phi_f0, _L, _a_list):
    _, phi_f_arr = fenics_to_numpy(_mesh, _phi_f)
    _, u_s_new_arr = fenics_to_numpy(_mesh, _u_s_new)
    _, u_s_old_arr = fenics_to_numpy(_mesh, _u_s_old)
    dus_dt_arr = (u_s_new_arr - u_s_old_arr) / delta_t
    dus_dxi_arr = np.gradient(u_s_new_arr, xi_arr)
    da_dt_val = (_a_list[-1] - _a_list[-2]) / delta_t
    return ((1 - _phi_f0) / (1 - phi_f_arr) *
            (dus_dt_arr - (1 - xi_arr) / (_L - _a_list[-1]) * da_dt_val * dus_dxi_arr))


v_s_ = Quantity("$v_{s}$", "YlOrBr", 7, mesh)
v_s_.f = vt
quantities = [phi_f, E, c, u_s, v_s_]


"""
Set up figure for the overall plot
"""
# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
Quantity.set_axs(quantities, axs)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves and save all our data
"""
# saving = [True, True, True, True]
# short_quants = ["phi", "E", "c", "u_s"]
saving = [True, True, True, True, True]
short_quants = ["phi", "E", "c", "u_s", "v_s"]
file_names = [f"data/initial/{q}_bM_{beta_E_num}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)

Quantity.plot_quantities(quantities, norm, 0.0, saving, file_names)

# define the time derivatives
dphi_dt = (phi_f.g - phi_f.g_old) / delta_t
dE_dt = (E.g - E.g_old) / delta_t
dc_dt = (c.g - c.g_old) / delta_t
da_dt = (a - a_old) / delta_t
dus_dt = (u_s.g - u_s.g_old) / delta_t

k_e.g = compute_k_e(phi_f.g, phi_f0, k_0, mu)
sigma_e.g = compute_sigma_e(phi_f.g, phi_f0, nu)

# Find intermediate expressions for the solid and fluid velocities
v_s = vt + phi_f.g * k_e.g * (E.g * sigma_e.g).dx(0) / ((L - a) * (1 - phi_f.g))
v_f = vt - k_e.g * (E.g * sigma_e.g).dx(0) / (L - a)

"""
Define the weak form
"""

# Weak form for the phi equation
Fun_phi = ((dphi_dt - da_dt * phi_f.g / (L - a)) * phi_f.v * dx +
           ((1 / (L - a))**2 * phi_f.g * k_e.g * (E.g * sigma_e.g).dx(0) -
            (1 / (L - a)) * phi_f.g * (vt - (1 - xi) * da_dt)) * phi_f.v.dx(0) * dx +
           (vt - (1 - xi) * da_dt) * phi_f.v / (L - a) * ds)

# Weak form for the E equation
Fun_E = (dE_dt + beta_E * c.g * E.g
         + (v_s - (1 - xi) * da_dt) / (L - a) * E.g.dx(0)) * E.v * dx
# Fun_E = dE_dt * E.v * dx + beta_E * c.g * E.g * E.v * dx

# Weak form for the c equation
Fun_c = ((phi_f.g * dc_dt + dphi_dt * c.g + da_dt * c.g *
          ((1 - xi) * phi_f.g.dx(0) - phi_f.g) / (L - a)) * c.v * dx +
         phi_f.g / (L - a) * (D_m * c.g.dx(0) / (L - a) - v_f * c.g) * c.v.dx(0) * dx)

# Weak form for the displacement
Fun_us = ((u_s.g.dx(0) * u_s.v -
           (phi_f.g - phi_f0) * (L - a) / (1 - phi_f0) * u_s.v) * dx)
# Fun_us = ((u_s.g * u_s.v.dx(0) +
#            (phi_f.g - phi_f0) / ((1 - phi_f0) * (L - a)) * u_s.v) * dx -
#           a * u_s.v * (1 - xi) * ds)

# Weak form for the moving boundary
Fun_a = (phi_f.g - 1 + (1 - phi_f0) / (1 - a / L)) * v_a * dx

# Combining the weak forms
Fun = Fun_phi + Fun_E + Fun_c + Fun_us + Fun_a


# Define the Jacobian, problem and solver
jacobian = derivative(Fun, w)
problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
solver = NonlinearVariationalSolver(problem)

"""
Loop over time steps and solve
"""
for n in range(N_time):
    print("Time:", n * delta_t)

    # Update some variables
    vt.t = n * delta_t

    # Solve
    solver.solve()
    phi_f.f, E.f, c.f, u_s_new, a_f = w.split(deepcopy=True)
    v_s_.f = get_vs_from_E_phi(mesh, phi_f.f, E.f, a_f(0.0), phi_f0_num,
                               k0_num, mu_num, nu_num, L_num)
    a_list.append(a_f(0.0))
    # v_s_.f = get_vs_from_u_phi(mesh, phi_f.f, u_s_new, u_s.f, phi_f0_num,
    #                            L_num, a_list)

    w_old.assign(w)

    # plot at the current timepoint
    Quantity.plot_quantities(quantities, norm, (n + 1) * delta_t, saving, file_names)
    u_s.f = u_s_new

"""
Colourbars
"""

# phi_f
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f.cmap),
             orientation='vertical',
             label='$t$', ax=phi_f.ax)
phi_f.label_plot(x_label="$\\xi$", title="Porosity")

# E
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=E.cmap),
             orientation='vertical',
             label='$t$', ax=E.ax)
E.label_plot(x_label="$\\xi$", title="Young's modulus")

# c
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c.cmap),
             orientation='vertical',
             label='$t$', ax=c.ax)
c.label_plot(x_label="$\\xi$", title="Solute concentration")

# u_s
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s.cmap),
             orientation='vertical',
             label='$t$', ax=u_s.ax)
u_s.label_plot(x_label="$\\xi$", title="Displacement")

# diff
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_s_.cmap),
             orientation='vertical',
             label='$t$', ax=v_s_.ax)
v_s_.label_plot(x_label="$\\xi$", title="Difference")

# Remove titles
for ax in axs:
    ax.set_title("")

# Save figure
fig.savefig(f"plots/initial/coupling_a/time_traces_v_0_01.png", bbox_inches="tight")

# Create figure for the left boundary over time
fig_a, ax_a = plt.subplots()
times = np.linspace(0, N_time * delta_t, N_time + 1)
ax_a.plot(np.array(a_list), times,
          color='darkviolet', label='$a(t)$')
ax_a.set_xlabel("Left boundary")
ax_a.set_ylabel("Time")
ax_a.legend()
# ax_a.set_xlim(min(a_list), max(a_list))
fig_a.savefig(f"plots/initial/coupling_a/left_bdry_v_0_01.png",
              bbox_inches="tight")
