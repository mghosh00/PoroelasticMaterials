"""
This Python code solves the following system for the solute concentration c_{0}
and the p-wave modulus \\mathcal{M}_{0}:

        \\diffp{c_{0}}{t} = \\mathrm{Pe}^{-1}\\diffp[2]{c_{0}}{x} - v_{f,0}\\diffp{c_{0}}{x},

        \\diffp{\\mathcal{M}_{0}}{t} = -b_{\\mathcal{M}}c_{0}\\mathcal{M}_{0},

for a known fluid velocity v_{f,0} = \\hat{Q}(t). We know this velocity as the fluid
velocity has a vanishing spatial gradient in this limit of large v_{f}.

The initial and boundary conditions are:

        \\mathrm{Pe}^{-1}\\diffp{c_{0}}{t}(0, t) - \\hat{Q}(t)c_{0}(0, t) = 0,
        \\mathrm{Pe}^{-1}\\diffp{c_{0}}{t}(1, t) - \\hat{Q}(t)c_{0}(1, t) = 0,
        c_{0}(x, 0) = 0,
        \\mathcal{M}_{0}(x, 0) = 1,

We will first solve the diffusion equation for c_{0}, followed by the degradation
equation for \\mathcal{M}_{0}.

We can then derive the first order porosity, \\phi_{f,1}, and the leading order
displacement, u_{x,0}, from the following relations:

\\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}},

\\diffp{u_{s,0}}{x} = \\frac{\\phi_{f,1}}{1 - \\phi_{f,0}}.

We already know the stress \\sigma_{xx,0}' in the following analytic form:

\\sigma_{xx,0}'(x, t) = \\phi_{f,0}\\hat{Q}(t)(g(t) - x),

where

g(t) = 1 - \\frac{2}{1 - \\phi_{f,0}}\\sqrt{\\gamma(t - 1)}{\\pi}

We will also prescribe various forms of the imposed volume flux, starting from
\\hat{Q}(t) = 1.

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

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(0.5)

# Compressive stress applied at the left boundary
sigma_prime_star = Constant(0.0)

# Degradation parameter
b_M = Constant(1.0)

# Péclet number
Pe = Constant(2.0)

b_M_num = float(b_M.values()[0])
Pe_num = float(Pe.values()[0])

# gamma and epsilon
epsilon = 0.01
gamma = Expression('exp(-bM) / epsilon', degree=1, bM=b_M, epsilon=epsilon)

"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 41

# Start time
ts = 0.0

# Number of mesh points
N_x = 40

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the x coodinates
x = mesh.coordinates()[:]

"""
Defining the driving flux
"""

# Imposed fluid flux - here we have a step function
# Qt = Expression('t < delta_t * N_time / 2 ? 0.0 : 1.0',
#                 degree=1, t=0.0, delta_t=delta_t, N_time=N_time)
Qt = Expression('t < delta_t * N_time / 2 ? 0.0 : 0.0',
                degree=1, t=ts, delta_t=delta_t, N_time=N_time)
# Qt = Expression('t < delta_t * N_time / 3 ? 0.0 : '
#                 't < 2 * delta_t * N_time / 3 ? 1.0 : 0.0',
#                 degree=1, t=0.0, delta_t=delta_t, N_time=N_time)
v_f0 = Qt

# Solution from the boundary layer matching
# gt = Expression('1 - (2 / (1 - phi_f0)) * sqrt(gamma * (t - 1) / pi)', degree=1,
#                 phi_f0=phi_f0, gamma=gamma, t=0.0)
gt = Expression('1 - (2 / (1 - phi_f0)) * sqrt(gamma * (t - 1) / pi)', degree=1,
                phi_f0=phi_f0, gamma=gamma, t=ts)

# Stress
sigma_xx0_prime_expr = Expression('phi_f0 * Qt * (gt - x[0])', degree=1, phi_f0=phi_f0, Qt=Qt, gt=gt)
sigma_xx0_prime = Quantity(r"$\sigma_{xx,0}'$", "YlOrBr",
                           2, mesh, sigma_xx0_prime_expr)

"""
Define the solutions c_0, M_0
"""

c_0 = Quantity(r"$c_{0}$", "Reds", 0, mesh)
M_0 = Quantity(r"$\mathcal{M}_{0}$", "Purples", 1, mesh)

"""
Define the Dirichlet boundary conditions
"""


# Define a function for the left boundary; this function just needs to return
# the value true when x is close to the boundary 0
def left(x):
    return near(x[0], 0)


# Define a function for the right boundary; this function just needs to return
# the value true when x is close to the boundary 1
def right(x):
    return near(x[0], 1)


# Define the boundary conditions at the left and right
# bc_left_c = DirichletBC(V, 0, left)
# bc_left_c = DirichletBC(V, 1, left)
# bc_right_c = DirichletBC(V, 1, right)

# Append the boundary conditions to the Quantities
# c_0.add_bc(bc_left_c)
# c_0.add_bc(bc_right_c)


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


def get_phi(_mesh: Mesh, _sigma_xx0_prime: Function, _M_0: Function, _phi_f0: Constant,
            _epsilon: float = 1.0):
    """Calculates the first order displacement given sigma_xx0_prime and
    phi_f0 according to the below formula:

    \\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}}.

    :param _mesh: The mesh for the domain.
    :param _sigma_xx0_prime: The given function for the effective stress.
    :param _M_0: The given function for the p-wave modulus.
    :param _phi_f0: The constant leading-order porosity.
    :param _epsilon: The epsilon parameter. This is for rescaling purposes.
    :return: The first-order porosity \\phi_{f,1}.
    """
    mesh_array, sigma_xx0_prime_array = fenics_to_numpy(_mesh, _sigma_xx0_prime)
    mesh_array, M_0_array = fenics_to_numpy(_mesh, _M_0)
    phi_f0_float = phi_f0.values()[0]
    # Calculate phi_f1 below
    phi_f1 = (1 - phi_f0_float) * sigma_xx0_prime_array / M_0_array
    return mesh_array.transpose()[0], phi_f1 * _epsilon


def get_u(_mesh: Mesh, _phi_f1: Function, _phi_f0: Constant, _epsilon: float = 1.0):
    """Calculates the leading order displacement given phi_f1 and phi_f0 according
    to the below formula:

    \\diffp{u_{s,0}}{x} = \\frac{\\phi_{f,1}}{1 - \\phi_{f,0}}

    This function uses scipy's cumtrapz function to approximate the indefinite
    integral. We also have the fixed boundary condition that u_s0(x=1) = 0,
    and so we will adjust the solution after integrating to ensure that u_s0
    is at 0 when x is 1.

    :param _mesh: The mesh for the domain.
    :param _phi_f1: The given function for first-order porosity.
    :param _phi_f0: The constant leading-order porosity.
    :param _epsilon: The epsilon parameter. This is for rescaling purposes.
    :return: The leading-order displacement u_{s,0}.
    """
    mesh_array, phi_f1_array = fenics_to_numpy(_mesh, _phi_f1)
    mesh_dx = mesh_array[1] - mesh_array[0]
    phi_f0_float = phi_f0.values()[0]
    u_s0_shifted = si.cumtrapz(phi_f1_array / _epsilon, dx=mesh_dx, initial=0) / (1 - phi_f0_float)
    # The below ensures u_s0 = 0 at x = 1
    u_s0 = u_s0_shifted - u_s0_shifted[-1]
    return mesh_array.transpose()[0], u_s0


"""
Define the initial condition
"""

# Expression for the initial conditions
c_0.bind_ic(Constant(1))
# c_0_ic = Expression('x[0]', degree=1)
M_0.bind_ic(Expression('exp(-bM * t)', degree=1, bM=b_M, t=ts))

# interpolate sigma
sigma_xx0_prime.interpolate()

# Define our porosity and displacement
_, phi_array = get_phi(mesh, sigma_xx0_prime.f, M_0.f, phi_f0)
x_coords, u_array = get_u(mesh, phi_array, phi_f0)

# phi_f1 = Quantity(r"$\epsilon\phi_{f,1}$", "Blues", 3, mesh)
phi_f1 = Quantity(r"$\phi_{f,1}$", "Blues", 3, mesh)
u_s0 = Quantity(r"$u_{s,0}$", "Greens", 4, mesh)
phi_f1.f = phi_array
u_s0.f = u_array

quantities = [c_0, M_0, sigma_xx0_prime, phi_f1, u_s0]

# Set up data saving
saving = [False, False, False, False, False]
short_quants = ["c", "M", "sigma", "phi", "us"]
file_names = [f"data_no_bl/Q_step_new/{q}_bM_{b_M_num}.csv" for q in short_quants]
# for file_name in file_names:
#     pd.DataFrame().to_csv(file_name)

"""
Set up figure for the overall plot
"""
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
# fig.subplots_adjust(bottom=0.5)
Quantity.set_axs(quantities, axs)
# norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)
norm = mpl.colors.Normalize(vmin=ts, vmax=ts + N_time * delta_t)

times = np.linspace(ts, ts + N_time * delta_t, N_time + 1)

"""
Plot the initial curves
"""
Quantity.plot_quantities(quantities, norm, ts, saving, file_names)

"""
Loop over time steps and solve
"""
for n in range(N_time):

    """
    Define the weak form
    """
    # define the time derivatives
    dc_0_dt = (c_0.g - c_0.g_old) / delta_t
    dM_0_dt = (M_0.g - M_0.g_old) / delta_t

    # Evaluate Qt, gt and sigma_xx0_prime at the current timepoint
    Qt.t = ts + n * delta_t
    gt.t = ts + n * delta_t
    sigma_xx0_prime.interpolate()

    # Weak form of c includes the Neumann boundary condition on x = 1
    # Fun_c = (dc_0_dt * v_c * dx + D_M_tilde * c_0.dx(0) * v_c.dx(0) * dx
    #          - D_M_tilde * v_c * ds)
    Fun_c = (dc_0_dt * c_0.v * dx + (1 / Pe) * c_0.g.dx(0) * c_0.v.dx(0) * dx
             - Qt * c_0.g * c_0.v.dx(0) * dx)
    # Weak form of M has no x derivatives
    Fun_M = dM_0_dt * M_0.v * dx + b_M * c_0.g * M_0.g * M_0.v * dx

    # solve the weak forms
    c_0.solve(Fun_c)
    M_0.solve(Fun_M)

    # record the porosity and displacement at this time point
    _, phi_array = get_phi(mesh, sigma_xx0_prime.f, M_0.f, phi_f0)
    x_coords, u_array = get_u(mesh, phi_array, phi_f0)
    phi_f1.f = phi_array
    u_s0.f = u_array

    # plot at the current timepoint
    Quantity.plot_quantities(quantities, norm, ts + n * delta_t, saving, file_names)


"""
Plot the solutions at the final time step and finalise the plots
"""
Quantity.plot_quantities(quantities, norm, ts + N_time * delta_t, saving, file_names)

# c_0
cb_c = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c_0.cmap),
                    orientation='vertical',
                    ax=c_0.ax)
cb_c.set_label(label=r'$t$', fontsize='xx-large')
c_0.label_plot("", label_size='xx-large')

# M_0
cb_M = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=M_0.cmap),
                    orientation='vertical',
                    ax=M_0.ax)
cb_M.set_label(label=r'$t$', fontsize='xx-large')
M_0.label_plot("", label_size='xx-large')

# sigma_xx,0'
cb_sigma = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=sigma_xx0_prime.cmap),
                        orientation='vertical',
                        ax=sigma_xx0_prime.ax)
cb_sigma.set_label(label=r'$t$', fontsize='xx-large')
sigma_xx0_prime.label_plot("", label_size='xx-large')
sigma_xx0_prime.ax.set_ylim(-2.5, 0.6)

"""
Plot the solution for the first-order porosity and leading-order displacement
"""
# phi_f,1
cb_phi = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1.cmap),
                      orientation='vertical',
                      ax=phi_f1.ax)
cb_phi.set_label(label=r'$t$', fontsize='xx-large')
phi_f1.label_plot("", label_size='xx-large')
# phi_f1.ax.set_ylim(-0.04, 0.01)
# phi_f1.ax.set_ylim(-0.1, 0.05)
phi_f1.ax.set_ylim(-4.0, 1.0)

# u_s,0
cb_u = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s0.cmap),
                    orientation='vertical',
                    ax=u_s0.ax)
cb_u.set_label(label=r'$t$', fontsize='xx-large')
u_s0.label_plot("", label_size='xx-large',
                x_label=r'$x$')
u_s0.ax.set_ylim(-1, 6)
# u_s0.ax.set_ylim(-1, 15)

"""
Save figure
"""
# fig.suptitle(f"FEniCS solution with $b_M={b_M_num}$")
fig.savefig(f"plots/linear_large_vf/fenics_b_M_{b_M_num}_Pe_{Pe_num}_eps_{epsilon}_x.png", bbox_inches="tight")
