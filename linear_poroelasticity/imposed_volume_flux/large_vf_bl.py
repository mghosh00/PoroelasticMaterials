"""
This Python code solves the following system for the fluid velocity within a time
boundary layer, v_{f,0}(x, T)

        e^{b_{\\mathcal{M}}}(1 - \\phi_{f,0})\\diffp{v_{f,0}}{T} = \\diffp[2]{v_{f,0}}{x},

which will give us then solutions for all other variables in the boundary layer region.

The initial and boundary conditions are:

        v_{f,0}(0, T) = 1, \\diffp{v_{f,0}}{x}(1, T) = 0,
        v_{f,0}(x, 0) = 0.

We can then derive the first order porosity, \\phi_{f,1}, the Terzaghi stress,
\\sigma_{xx,0}', and the leading order displacement, u_{s,0}, from the following
relations:

\\phi_{f,0}v_{f,0} = -\\diffp{\\sigma_{xx,0}'}{x}

\\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}},

\\diffp{u_{s,0}}{x} = \\frac{\\phi_{f,1}}{1 - \\phi_{f,0}}.

"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as si
import pandas as pd

from quantity import Quantity

"""
Define model parameters
"""

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(0.5)

# Degradation parameter
b_M_val = 3.0
b_M = Constant(b_M_val)

# Péclet number
Pe = Constant(2.0)

# Constant (in this regime) p-wave modulus
M_0 = Constant(np.exp(-b_M_val))

# No stress on the right, no porosity initially and no displacement on the right
b_M_num = float(b_M.values()[0])
Pe_num = float(Pe.values()[0])

"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 200

# Number of mesh points
N_x = 40

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the x coodinates
x = mesh.coordinates()[:]

"""
Define the solution v_f0
"""

v_f0 = Quantity("$v_{f,0}$", "PuRd", 0, mesh)

"""
Define the Dirichlet boundary conditions
"""


# Define a function for the left boundary; this function
# just needs to return the value true when x is close to
# the boundary 0
def left(x):
    return near(x[0], 0)


# Define a function for the right boundary; this function
# just needs to return the value true when x is close to
# the boundary 1
def right(x):
    return near(x[0], 1)


# Define the boundary conditions at the left and right
bc_left_v = DirichletBC(v_f0.V, 1, left)

# Append the boundary condition to the Quantity
v_f0.add_bc(bc_left_v)


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


def get_sigma(_mesh: Mesh, _v_f0: Function, _phi_f0: Constant):
    """Calculates the Terzaghi stress according the differential equation:

    \\diffp{\\sigma_{xx,0}'}{x} = -\\phi_{f,0}v_{f,0}
    \\sigma_{xx,0}'(1, T) = 0

    This function uses scipy's cumtrapz function to approximate the indefinite
    integral.

    :param _mesh: The mesh for the domain.
    :param _v_f0: The given function for leading-order fluid velocity.
    :param _phi_f0: The constant leading-order porosity.
    :return: The leading-order Terzaghi stress \\sigma_{xx,0}'.
    """
    mesh_array, v_f0_array = fenics_to_numpy(_mesh, _v_f0)
    mesh_dx = mesh_array[1] - mesh_array[0]
    phi_f0_float = phi_f0.values()[0]
    sigma_xx0_prime_shifted = - phi_f0_float * si.cumtrapz(v_f0_array, dx=mesh_dx, initial=0)
    # The below ensures sigma_xx0' = 0 at x = 1
    sigma_xx0_prime = sigma_xx0_prime_shifted - sigma_xx0_prime_shifted[-1]
    return mesh_array.transpose()[0], sigma_xx0_prime


def get_phi(_mesh: Mesh, _sigma_xx0_prime: Function, _M_0: Constant, _phi_f0: Constant):
    """Calculates the first order displacement given sigma_xx0_prime and
    phi_f0 according to the below formula:

    \\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}}.

    :param _mesh: The mesh for the domain.
    :param _sigma_xx0_prime: The given function for the effective stress.
    :param _M_0: The given fixed p-wave modulus.
    :param _phi_f0: The constant leading-order porosity.
    :return: The first-order porosity \\phi_{f,1}.
    """
    mesh_array, sigma_xx0_prime_array = fenics_to_numpy(_mesh, _sigma_xx0_prime)
    M_0_float = _M_0.values()[0]
    phi_f0_float = phi_f0.values()[0]
    # Calculate phi_f1 below
    phi_f1 = (1 - phi_f0_float) * sigma_xx0_prime_array / M_0_float
    return mesh_array.transpose()[0], phi_f1


def get_u(_mesh: Mesh, _phi_f1: Function, _phi_f0: Constant):
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
    :return: The leading-order displacement u_{s,0}.
    """
    mesh_array, phi_f1_array = fenics_to_numpy(_mesh, _phi_f1)
    mesh_dx = mesh_array[1] - mesh_array[0]
    phi_f0_float = phi_f0.values()[0]
    u_s0_shifted = si.cumtrapz(phi_f1_array, dx=mesh_dx, initial=0) / (1 - phi_f0_float)
    # The below ensures u_s0 = 0 at x = 1
    u_s0 = u_s0_shifted - u_s0_shifted[-1]
    return mesh_array.transpose()[0], u_s0


"""
Define the initial condition
"""

# Expression for the initial conditions
v_f0.bind_ic(Constant(0))

# Define Quantities sigma, phi and u
_, sigma_array = get_sigma(mesh, v_f0.f, phi_f0)
_, phi_array = get_phi(mesh, sigma_array, M_0, phi_f0)
x_coords, u_array = get_u(mesh, phi_array, phi_f0)

sigma_xx0 = Quantity("$\\sigma_{xx,0}'$", "YlOrBr", 1, mesh)
phi_f1 = Quantity("$\\phi_{f,1}$", "Blues", 2, mesh)
u_s0 = Quantity("$u_{s,0}$", "Greens", 3, mesh)
sigma_xx0.f = sigma_array
phi_f1.f = phi_array
u_s0.f = u_array

quantities = [v_f0, sigma_xx0, phi_f1, u_s0]

# Set up data saving
saving = [True, True, True, True]
short_quants = ["vf", "sigma", "phi", "us"]
file_names = [f"data_bl/Q_step/{q}_bM_{b_M_num}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)


"""
Set up figure for the overall plot
"""
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(4, 40/3), sharex=True)
# fig.subplots_adjust(bottom=0.5)
Quantity.set_axs(quantities, axs)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves
"""
Quantity.plot_quantities(quantities, norm, 0.0, saving, file_names)

"""
Loop over time steps and solve
"""
for n in range(N_time):

    """
    Define the weak form
    """
    # define the time derivatives
    dv_f0_dt = (v_f0.f - v_f0.f_old) / delta_t

    # Weak form of v includes the Neumann boundary condition on x = 1
    Fun_v = M_0 * (1 - phi_f0) * dv_f0_dt * v_f0.v * dx + v_f0.f.dx(0) * v_f0.v.dx(0) * dx

    v_f0.solve(Fun_v)

    # record the stress, porosity and displacement at this time point
    _, sigma_array = get_sigma(mesh, v_f0.f, phi_f0)
    _, phi_array = get_phi(mesh, sigma_array, M_0, phi_f0)
    x_coords, u_array = get_u(mesh, phi_array, phi_f0)
    sigma_xx0.f = sigma_array
    phi_f1.f = phi_array
    u_s0.f = u_array

    # plot at this timepoint
    Quantity.plot_quantities(quantities, norm, n * delta_t, saving, file_names)


"""
Plot the solutions at the final time step and finalise the plots
"""

Quantity.plot_quantities(quantities, norm, N_time * delta_t, saving, file_names)

# v_f,0
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_f0.cmap),
             orientation='vertical',
             label='$t$', ax=v_f0.ax)
v_f0.label_plot("Leading order fluid velocity")

# sigma_xx,0'
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=sigma_xx0.cmap),
             orientation='vertical',
             label='$t$', ax=sigma_xx0.ax)
sigma_xx0.label_plot("Leading order Terzaghi stress")

# phi_f,1
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1.cmap),
             orientation='vertical',
             label='$t$', ax=phi_f1.ax)
phi_f1.label_plot("First order porosity")

# u_s,0
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s0.cmap),
             orientation='vertical',
             label='$t$', ax=u_s0.ax)
u_s0.label_plot("Leading order displacement")

# plt.show()
fig.suptitle(f"FEniCS solution with $b_M={b_M_num}$ within boundary layer")
fig.savefig(f"plots/Q_step_bl/fenics_b_M_{b_M_num}_Pe_{Pe_num}.png", bbox_inches="tight")
