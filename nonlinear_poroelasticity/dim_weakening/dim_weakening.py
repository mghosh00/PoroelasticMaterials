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
        t.b.d.

The moving boundary can be determined by the following implicit relation:

        a(t) = \\phi_{f,0}L - \\int_{a(t)}^{L}\\phi_{f}(x, t)dx,

given a known profile for \\phi_{f} at the previous timestep (in the numerical scheme).
We will also change coordinates onto a fixed domain (see details below).
"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as si
import pandas as pd

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
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
c_star = Constant(1)

x = Expression('x[0]', degree=1)

L_num = float(L.values()[0])
phi_f0_num = float(phi_f0.values()[0])
beta_E_num = float(beta_E.values()[0])
D_m_num = float(D_m.values()[0])

# Setting up the moving boundary
a = [0.0]

# IMPORTANT!! The below line is only true when the system starts off with no
# deformation (which is the case when we have prescribed v = 0 initially)
a_dot = [0.0]

"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 41

# Number of mesh points
N_x = 40

# Imposed phase-averaged velocity
vt = Expression('t < delta_t * N_time / 2 ? 0.0 : 0.0',
                degree=1, t=0.0, delta_t=delta_t, N_time=N_time)

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the xi coodinates
xi = Expression('x[0]', degree=1)

# Set up function space
P1 = FiniteElement("Lagrange", interval, 1)
element = MixedElement([P1, P1, P1])
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

# Set up the functions from the joint space
v_phi, v_E, v_c = TestFunctions(V)

# Define the initial conditions
w_0 = Expression(('phi_f0', '1', '1'), degree=1, phi_f0=phi_f0)
w_old = project(w_0, V)


w = Function(V)
w_phi, w_E, w_c = split(w)
phi_f.f, E.f, c.f = w_old.split(deepcopy=True)
w_phi_old, w_E_old, w_c_old = split(w_old)
phi_f.set_sym_functions(w_phi, v_phi, w_phi_old)
E.set_sym_functions(w_E, v_E, w_E_old)
c.set_sym_functions(w_c, v_c, w_c_old)

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


k_e = Quantity("$k_{e}(\\phi_{f})$", "GnBu", 3, mesh)
sigma_e = Quantity("$\\sigma_{e}(\\phi_{f})$", "YlOrBr", 4, mesh)

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
bcs = []


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


def get_us(_mesh: Mesh, _phi_f: Function, _L: float, _a_t: float, _phi_f0: Constant):
    """Calculates the leading order displacement given phi_f1 and phi_f0 according
    to the below formula:

    \\diffp{u_s}{\\xi} = \\frac{\\phi_{f} - \\phi_{f,0}}{(1 - \\phi_{f,0})(L - a(t))}

    This function uses scipy's cumtrapz function to approximate the indefinite
    integral. We also have the fixed boundary condition that u_s(x=L) = 0,
    and so we will adjust the solution after integrating to ensure that u_s0
    is at 0 when \\xi is 1.

    :param _mesh: The mesh for the domain.
    :param _phi_f: The given function for porosity.
    :param _L: The length of the domain.
    :param _a_t: The moving boundary at current time.
    :param _phi_f0: The constant leading-order porosity.
    :return: The displacement u_s.
    """
    mesh_array, phi_f_array = fenics_to_numpy(_mesh, _phi_f)
    mesh_dx = mesh_array[1] - mesh_array[0]
    phi_f0_float = phi_f0.values()[0]
    u_s0_shifted = ((si.cumtrapz(phi_f_array, dx=mesh_dx, initial=0) - phi_f0_float) /
                    ((1 - phi_f0_float) * (_L - _a_t)))
    # The below ensures u_s0 = 0 at x = 1
    u_s0 = u_s0_shifted - u_s0_shifted[-1]
    return mesh_array.transpose()[0], u_s0


def get_a(_mesh: Mesh, _phi_f: Function, _L: Constant, _phi_f0: Constant):
    """Calculates the left moving boundary given a porosity profile phi_f as:
    a(t) = L - \\frac{1 - \\phi_{f,0}}{1 - \\int_{0}^{1}\\phi_{f}(\\xi, t)d\\xi}.
    This function uses scipy's simpson function to find the indefinite integral.

    :param _mesh: The mesh for the domain.
    :param _phi_f: The current porosity profile in (xi, t) coordinates.
    :param _L: The length of the domain.
    :param _phi_f0: The constant initial porosity.
    :return: The current position of the left boundary.
    """
    L_val = _L.values()[0]
    phi_f0_val = _phi_f0.values()[0]
    integral_of_phi = assemble(_phi_f)
    a_t = L_val * (1 - (1 - phi_f0_val) / (1 - integral_of_phi))
    return a_t


# interpolate([phi_f0, Constant(1), Constant(1)], V)
# u.interpolate([phi_f0, Constant(1), Constant(1)])
# phi_f.bind_ic(phi_f0)
# c.bind_ic(Constant(1))
# E.bind_ic(Constant(1))

# Define our displacement
x_coords, u_array = get_us(mesh, phi_f.f, L_num, float(a[0]), phi_f0)
u_s = Quantity("$u_s$", "Greens", 5, mesh)
u_s.f = u_array

# Define our fluid and solid velocities on the right
# qt = 0
# _, v_s0_array = get_vs_R(mesh, phi_f1_R.f, x_c[0], phi_f0, qt, D_phi)
# _, v_f0_array = get_vf_R(mesh, v_s0_array, phi_f0, qt)
# v_s0_R = Quantity("$v_{s,0}^{R}$", "Oranges", 6, mesh)
# v_f0_R = Quantity("$v_{f,0}^{R}$", "PuRd", 7, mesh)
# v_s0_R.f = v_s0_array
# v_f0_R.f = v_f0_array

quantities = [phi_f, E, c, u_s]


"""
Set up figure for the overall plot
"""
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
# fig.subplots_adjust(bottom=0.5)
Quantity.set_axs(quantities, axs)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves and save all our data
"""
saving = [True, True, True, True]
short_quants = ["phi", "E", "c", "u_s"]
file_names = [f"data/initial/{q}_bM_{beta_E_num}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)

Quantity.plot_quantities(quantities, norm, 0.0, saving, file_names)

"""
Loop over time steps and solve
"""
for n in range(N_time):
    print("Time: ", n * delta_t)
    a_dot_n = float(a_dot[n])
    a_n = float(a[n])

    # Update some variables
    k_e.g = compute_k_e(phi_f.g, phi_f0, k_0, mu)
    sigma_e.g = compute_sigma_e(phi_f.g, phi_f0, nu)
    vt.t = n * delta_t

    """
    Define the weak form
    """
    # define the time derivatives
    dphi_dt = (phi_f.g - phi_f.g_old) / delta_t
    dE_dt = (E.g - E.g_old) / delta_t
    dc_dt = (c.g - c.g_old) / delta_t

    # Find intermediate expressions for the solid and fluid velocities
    v_s = vt + phi_f.g * k_e.g * (E.g * sigma_e.g).dx(0) / ((L - a_n) * (1 - phi_f.g))
    v_f = vt - k_e.g * (E.g * sigma_e.g).dx(0) / (L - a_n)

    # Weak form for the phi equation
    Fun_phi = ((dphi_dt - a_dot_n * phi_f.g / (L - a_n)) * phi_f.v * dx +
               ((1 / (L - a_n))**2 * phi_f.g * k_e.g * (E.g * sigma_e.g).dx(0) -
               (1 / (L - a_n)) * phi_f.g * (vt - (1 - xi) * a_dot_n)) * phi_f.v.dx(0) * dx +
               (vt - (1 - xi) * a_dot_n) * phi_f.v / (L - a_n) * ds)

    # Weak form for the E equation
    Fun_E = (dE_dt + beta_E * c.g * E.g
             + (v_s - (1 - xi) * a_dot_n) / (L - a_n) * E.g.dx(0)) * E.v * dx
    # Fun_E = dE_dt * E.v * dx + beta_E * c.g * E.g * E.v * dx

    # Weak form for the c equation
    Fun_c = ((phi_f.g * dc_dt + dphi_dt * c.g + a_dot_n * c.g *
              ((1 - xi) * phi_f.g.dx(0) - phi_f.g) / (L - a_n)) * c.v * dx +
             phi_f.g / (L - a_n) * (D_m * c.g.dx(0) / (L - a_n) - v_f * c.g) * c.v.dx(0) * dx)

    # solve the joint weak form
    Fun = Fun_phi + Fun_E + Fun_c
    # Define the Jacobian, problem and solver
    jacobian = derivative(Fun, w)
    problem = NonlinearVariationalProblem(Fun, w, bcs, jacobian)
    solver = NonlinearVariationalSolver(problem)
    # solver.parameters['nonlinear_solver'] = 'newton'
    # sprms = solver.parameters['newton_solver']
    # sprms['maximum_iterations'] = 100

    # Solve the problem
    solver.solve()
    # solve(Fun_phi + Fun_E + Fun_c == 0, u, bcs)

    # plot at the current timepoint
    phi_f.f, E.f, c.f = w.split(deepcopy=True)

    # record the displacement at this time point
    x_coords, u_s_array = get_us(mesh, phi_f.f, L_num, float(a[n]), phi_f0)
    u_s.f = u_s_array
    Quantity.plot_quantities(quantities, norm, (n + 1) * delta_t, saving, file_names)

    w_old.assign(w)
    # Update the value of the left boundary
    a.append(get_a(mesh, phi_f.f, L, phi_f0))
    a_dot.append(a[n + 1] - a[n] / delta_t)

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

# Save figure
fig.suptitle(f"FEniCS solution with $\\beta_E={beta_E_num}$, $D_m={D_m_num}$")
fig.savefig(f"plots/initial/fenics_beta_E_{beta_E_num}_D_m_{D_m_num}_nocflux.png", bbox_inches="tight")

# Create figure for the left boundary over time
fig_a, ax_a = plt.subplots()
ax_a.plot(np.array(a), np.linspace(0, N_time * delta_t, N_time + 1), color='darkviolet')
ax_a.set_xlabel("Left boundary")
ax_a.set_ylabel("Time")
ax_a.set_xlim(0, L_num)
fig_a.suptitle(f"Left boundary over time with $\\beta_E={beta_E_num}$")
fig_a.savefig(f"plots/initial/left_bdry_beta_E_{beta_E_num}_D_m_{D_m_num}_nocflux.png", bbox_inches="tight")
