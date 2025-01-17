"""
This Python code solves the following system for the solute concentration c_{0},
the p-wave modulus \\mathcal{M}_{0} and the Terzaghi effective stress \\sigma_{xx,0}'

        \\diffp{c_{0}}{t} = \\tilde{\\mathcal{D}}_{m}\\diffp[2]{c_{0}}{x},

        \\diffp{\\mathcal{M}_{0}}{t} = -b_{\\mathcal{M}}c_{0}\\mathcal{M}_{0},

        \\diffp{\\sigma_{xx,0}'}{t} + b_{\\mathcal{M}}c_{0}\\sigma_{xx,0}' =
        \\mathcal{M}_{0}\\diffp[2]{\\sigma_{xx,0}'}{x}.

The initial and boundary conditions are:

        c_{0}(0, t) = 0,
        \\diffp{c_{0}}{t}(1, t) = 1,
        c_{0}(x, 0) = 1,
        \\mathcal{M}_{0}(x, 0) = 1,
        \\sigma_{xx,0}'(0, t) = \\sigma'^{*},
        \\sigma_{xx,0}'(1, t) = \\sigma'^{*}-\\Delta p,
        \\sigma_{xx,0}'(x, 0) = 0.

We will first solve the diffusion equation for c_{0}, followed by the degradation
equation for \\mathcal{M}_{0}, and then finally the diffusion equation for the
stress \\sigma_{xx,0}'.

We can then derive the first order porosity, \\phi_{f,1}, and the leading order
displacement, u_{x,0}, from the following relations:

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
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

"""
Define model parameters
"""

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(0.5)

# Compressive stress applied at the left boundary
sigma_prime_star = Constant(-0.5)

# Pressure difference across the domain
Delta_p = Constant(0.5)

# Degradation parameter
b_M = Constant(2.0)

# Diffusive parameter for the solute concentration
D_M_tilde = Constant(0.5)

b_M_num = float(b_M.values()[0])
D_m_num = float(D_M_tilde.values()[0])

"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 300

# Number of mesh points
N_x = 40

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the x coodinates
x = mesh.coordinates()[:]

"""
Define the solutions c_0, M_0 and sigma_xx0'
"""

c_0 = Quantity("$c_{0}$", "Reds", 0, mesh)
M_0 = Quantity("$\\mathcal{M}_{0}$", "Purples", 1, mesh)
sigma_xx0_prime = Quantity("$\\sigma_{xx,0}'$", "YlOrBr", 2, mesh)

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
bc_left_c = DirichletBC(c_0.V, 0, left)
# bc_left_c = DirichletBC(c_0.V, 1, left)
# bc_right_c = DirichletBC(c_0.V, 1, right)
bc_left_sigma = DirichletBC(sigma_xx0_prime.V, sigma_prime_star, left)
bc_right_sigma = DirichletBC(sigma_xx0_prime.V, sigma_prime_star - Delta_p, right)

# Append the bcs to the Quantities
c_0.add_bc(bc_left_c)
# c_0.add_bc(bc_right_c)
sigma_xx0_prime.add_bc(bc_left_sigma)
sigma_xx0_prime.add_bc(bc_right_sigma)


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


def get_phi(_mesh: Mesh, _sigma_xx0_prime: Function, _M_0: Function, _phi_f0: Constant):
    """Calculates the first order displacement given sigma_xx0_prime and
    phi_f0 according to the below formula:

    \\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}}.

    :param _mesh: The mesh for the domain.
    :param _sigma_xx0_prime: The given function for the effective stress.
    :param _M_0: The given function for the p-wave modulus.
    :param _phi_f0: The constant leading-order porosity.
    :return: The first-order porosity \\phi_{f,1}.
    """
    mesh_array, sigma_xx0_prime_array = fenics_to_numpy(_mesh, _sigma_xx0_prime)
    mesh_array, M_0_array = fenics_to_numpy(_mesh, _M_0)
    phi_f0_float = phi_f0.values()[0]
    # Calculate phi_f1 below
    phi_f1 = (1 - phi_f0_float) * sigma_xx0_prime_array / M_0_array
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
c_0.bind_ic(Constant(1))
# c_0.bind_ic(Expression('x[0]', degree=1))
M_0.bind_ic(Constant(1))
sigma_xx0_prime.bind_ic(Constant(0))

# Define our porosity and displacement
_, phi_array = get_phi(mesh, sigma_xx0_prime.f, M_0.f, phi_f0)
x_coords, u_array = get_u(mesh, phi_array, phi_f0)
phi_f1 = Quantity("$\\phi_{f,1}$", "Blues", 3, mesh)
u_s0 = Quantity("$u_{s,0}$", "Greens", 4, mesh)
phi_f1.f = phi_array
u_s0.f = u_array
quantities = [c_0, M_0, sigma_xx0_prime, phi_f1, u_s0]


"""
Set up figure for the overall plot
"""
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
# fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(4, 60/3), sharex=True)
# fig.subplots_adjust(bottom=0.5)
Quantity.set_axs(quantities, axs)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves and save all our data
"""
saving = [True, True, True, True, True]
# saving = [False, False, False, False, False]
short_quants = ["c", "M", "sigma", "phi", "u"]
file_names = [f"data/c0_diff_no_flux/{q}_bM_{b_M_num}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)

Quantity.plot_quantities(quantities, norm, 0.0, saving, file_names)

"""
Loop over time steps and solve
"""
for n in range(N_time):

    """
    Define the weak form
    """
    # define the time derivatives
    dc_0_dt = (c_0.f - c_0.f_old) / delta_t
    dM_0_dt = (M_0.f - M_0.f_old) / delta_t
    dsigma_xx0_prime_dt = (sigma_xx0_prime.f - sigma_xx0_prime.f_old) / delta_t

    # Weak form of c includes the Neumann boundary condition on x = 1
    # Fun_c = (dc_0_dt * c_0.v * dx + D_M_tilde * c_0.f.dx(0) * c_0.v.dx(0) * dx
    #          - D_M_tilde * c_0.v * ds)
    Fun_c = dc_0_dt * c_0.v * dx + D_M_tilde * c_0.f.dx(0) * c_0.v.dx(0) * dx
    # Weak form of M has no x derivatives
    Fun_M = dM_0_dt * M_0.v * dx + b_M * c_0.f * M_0.f * M_0.v * dx
    # Weak form of sigma_xx0_prime includes c_0 and M_0 in the definition
    Fun_sigma = ((dsigma_xx0_prime_dt + b_M * c_0.f * sigma_xx0_prime.f) *
                 sigma_xx0_prime.v * dx + (M_0.f * sigma_xx0_prime.v).dx(0) *
                 sigma_xx0_prime.f.dx(0) * dx)

    # solve the weak forms
    c_0.solve(Fun_c)
    M_0.solve(Fun_M)
    sigma_xx0_prime.solve(Fun_sigma)

    # record the porosity and displacement at this time point
    _, phi_array = get_phi(mesh, sigma_xx0_prime.f, M_0.f, phi_f0)
    x_coords, u_array = get_u(mesh, phi_array, phi_f0)
    phi_f1.f = phi_array
    u_s0.f = u_array

    # plot at the current timepoint
    Quantity.plot_quantities(quantities, norm, (n + 1) * delta_t, saving, file_names)

# record the solid velocity
# v_s0 = Quantity("$v_{s,0}$", "Oranges", 5, mesh)
# v_s0.set_ax(axs[5])
# u_s0_array = (pd.read_csv(f"data/c0_diff_no_flux/u_bM_{b_M_num}.csv", index_col=0)
#               .to_numpy())
# v_s0_array = np.gradient(u_s0_array, axis=1) / delta_t
# pd.DataFrame().to_csv(f"data/c0_diff_no_flux/vs_bM_{b_M_num}.csv")
# for n in range(u_s0_array.shape[1]):
#     v_s0.f = v_s0_array[:, n]
#     v_s0.plot(norm, n * delta_t, True, f"data/c0_diff_no_flux/vs_bM_{b_M_num}.csv")

"""
Plot the solutions at the final time step and finalise the plots
"""
Quantity.plot_quantities(quantities, norm, N_time * delta_t)
# v_s0.plot(norm, N_time * delta_t)


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

"""
Plot the solution for the first-order porosity and leading-order displacement
"""
# phi_f,1
cb_phi = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1.cmap),
                      orientation='vertical',
                      ax=phi_f1.ax)
cb_phi.set_label(label=r'$t$', fontsize='xx-large')
phi_f1.label_plot("", label_size='xx-large')

# u_s,0
cb_u = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s0.cmap),
                    orientation='vertical',
                    ax=u_s0.ax)
cb_u.set_label(label=r'$t$', fontsize='xx-large')
u_s0.label_plot("", label_size='xx-large',
                x_label=r'$x$')

# Set up limits - analytic c
# c_0.ax.set_ylim(-0.1, 1.1)
# u_s0.ax.set_ylim(0, 8.5)

# Set up limits - diffusive c
M_0.ax.set_ylim(0.0, 1.0)
phi_f1.ax.set_ylim(-3.5, 0.0)
u_s0.ax.set_ylim(0.0, 3.5)

# Save figure
# fig.suptitle(f"FEniCS solution with $b_M={b_M_num}$")
# fig.suptitle(f"FEniCS solution with $b_M={b_M_num}$")
fig.savefig(f"plots/c0_diffusion_sigma/Pe_O1_no_flux/fenics_b_M_{b_M_num}_D_m_{D_m_num}.png",
            bbox_inches='tight')
# fig.savefig(f"plots/c0_diffusion/fenics_b_M_{b_M_num}.png", bbox_inches="tight")
