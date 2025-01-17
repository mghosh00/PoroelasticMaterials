"""
This Python code solves the following system for the Terzaghi effective stress
\\sigma_{xx,0}' after time t = t_c

        \\diffp{\\sigma_{xx,0}'}{t} + b_{\\mathcal{M}}c_{0}\\sigma_{xx,0}' =
        \\mathcal{M}_{0}\\diffp[2]{\\sigma_{xx,0}'}{x},

for known, uncoupled solutions c_{0} and \\mathcal{M}_{0} (see degradation).

The initial and boundary conditions are:

        \\sigma_{xx,0}'(0, t) = \\sigma'^{*},
        \\sigma_{xx,0}'(x_{c}(t), t) = 0,
        \\sigma_{xx,0}'(x, 0) = \\hat{\\sigma}(x),

where \\hat{\\sigma}(x) is the solution at t = t_c (known).

We can then derive the first order porosity, \\phi_{f,1}, and the leading order
displacement, u_{x,0}, from the following relations:

        \\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}},

        \\diffp{u_{s,0}}{x} = \\frac{\\phi_{f,1}}{1 - \\phi_{f,0}}.

The above is the solution for p-wave modulus \\mathcal{M}_{0} \\geq \\mathcal{M}_{c},
but below this critical value, we postulate that the solid skeleton no longer behaves
elastically and degrades (dissolves) into the fluid. In this region, bounded by the
critical point x_{c}(t) s.t. \\mathcal{M}_{0}(x_{c}(t), t) = \\mathcal{M}_{c}, we
model the skeleton combined with fluid as a two-phase flow using drift-flux equations.
In this region we solve the equations

        \\diffp{q_{0}}{x} = 0,
        \\diffp{\\phi_{f,1}}{t} = \\mathcal{D}_{\\phi}\\diffp[2]{\\phi_{f,1}}{x},
        \\diffp{\\phi_{f,1}}{t} - (1 - \\phi_{f,0})\\diffp{v_{s,0}}{x} = 0,
        (1 - \\phi_{f,0})v_{s,0} = (1 - \\phi_{f,0})q_{0} + \\mathcal{D}_{\\phi}\\diffp{\\phi_{f,1}}{x}

subject to the following conditions

        v_{s,0}(1, t) = 0, (solid particles cannot travel through right)
        \\phi_{f,1}(x_{c}(t), t) = 0, (continuity of porosity across middle boundary)
        \\phi_{f,1}(1, t_c) = \\frac{1 - \\phi_{f,0}}{\\mathcal{M}_{c}}(\\sigma'^{*} - \\Delta p)

Note that we will solve all the above equations with a change of variables for the numerical
scheme, so that there is no moving boundary.
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

# Compressive stress applied at the left boundary
sigma_prime_star = Constant(-0.5)

# Pressure difference across the domain
Delta_p = Constant(0.5)

# Degradation parameter
b_M = Constant(2.0)

# Diffusive parameter for the solute concentration
D_M_tilde = Constant(0.5)

# Critical p-wave modulus (between 0 and 1)
M_c = Constant(0.7)

# Diffusive parameter for the porosity
D_phi = Constant(1.0)

# Porosity at the critical point
phi_c = Expression('(1 - phi_f0) * (sigma_prime_star - Delta_p) / M_c',
                   degree=1, phi_f0=phi_f0, sigma_prime_star=sigma_prime_star,
                   Delta_p=Delta_p, M_c=M_c)

# Flux, not sure of solution at this point
Q = Constant(1.0)
qt = Expression('Q', degree=1, Q=Q, t=0.0)
x = Expression('x[0]', degree=1)

b_M_num = float(b_M.values()[0])
D_m_num = float(D_M_tilde.values()[0])
M_c_num = float(M_c.values()[0])

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
Get the solution for the p-wave modulus and find the critical values
"""
M_0_array = (pd.read_csv(f"x_data/c0_diff_no_flux/M_bM_{b_M_num}.csv", index_col=0)
             .to_numpy())
c_0_array = (pd.read_csv(f"x_data/c0_diff_no_flux/c_bM_{b_M_num}.csv", index_col=0)
             .to_numpy())
sigma_array = (pd.read_csv(f"x_data/c0_diff_no_flux/sigma_bM_{b_M_num}.csv", index_col=0)
               .to_numpy())
# The numpy solution for x_c(t) and t_c
x_c = np.argmax(np.where(M_0_array < M_c_num, M_0_array, 0.0), axis=0) / N_x
i_c, t_c = np.argmax(x_c), np.argmax(x_c) * delta_t
# Now we convert to new time coordinates tau = t - t_c
x_c = x_c[i_c - 1:]
x_c[0] = 1.0
x_c_dot = np.gradient(x_c) / delta_t
N_time = N_time - i_c + 1

# To avoid later computational errors (should not make a different to the calculations)
x_c[0] = 0.999
x_c_dot += 1e-7

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the xi coodinates
xi = mesh.coordinates()[:]

"""
Function to change spatial coordinates and get to the correct mesh.
"""


def x_t_to_xi_L_tau(f: np.array, x_c: np.array, i_c: int):
    """Change the quantity from (x, t) coordinates to (xi_L, tau) where
    \\xi_{L} = x / x_{c}(t), \\tau = t - t_{c}. We also fit onto the new
    mesh, which will involve some interpolation.

    :param f: Some quantity in (x, t) coordinates (np array).
    :param x_c: The boundary curve x_{c}(t).
    :param i_c: The index for which t = t_{c} in this array.
    :return: The new array in (xi_L, tau) coordinates.
    """
    # First, shift the time variable (including timepoint just before region split)
    f = f[:, i_c - 1:]
    f_new = []

    # Now focus on each timestep
    for i in range(f.shape[1]):
        # Points left of x_c only
        f_i_left = f[:int(x_c[i] * N_x), i]
        # Stretch region to [0, 1] (using transformation) and interpolate back onto
        # an N_x grid
        f_i_left = np.interp(np.linspace(0, 1, N_x + 1),
                             np.linspace(0, 1, len(f_i_left)),
                             f_i_left)
        f_new.append(f_i_left)
    return np.array(f_new).transpose()


"""
Define the solutions c_0, M_0 and sigma_xx0' on the left and phi_f1 on the right
"""

c_0_L = Quantity("$c_{0}^{L}$", "Reds", 0, mesh)
M_0_L = Quantity("$\\mathcal{M}_{0}^{L}$", "Purples", 1, mesh)
sigma_xx0_L = Quantity("$\\sigma_{xx,0}^{L}'$", "YlOrBr", 2, mesh)
phi_f1_R = Quantity("$\\phi_{f,1}^{R}$", "Blues", 3, mesh)

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
bc_left_sigma_L = DirichletBC(sigma_xx0_L.V, sigma_prime_star, left)
bc_right_sigma_L = DirichletBC(sigma_xx0_L.V, 0, right)

bc_left_phi_R = DirichletBC(phi_f1_R.V, 0, left)

# Append the bcs to the Quantities
sigma_xx0_L.add_bc(bc_left_sigma_L)
sigma_xx0_L.add_bc(bc_right_sigma_L)
phi_f1_R.add_bc(bc_left_phi_R)


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


def get_phi_L(_mesh: Mesh, _sigma_xx0_L: Function, _M_0_L: Function, _phi_f0: Constant):
    """Calculates the first order displacement given sigma_xx0_prime and
    phi_f0 according to the below formula:

    \\phi_{f,1} = (1-\\phi_{f,0})\\frac{\\sigma_{xx,0}'}{\\mathcal{M}_{0}}.

    :param _mesh: The mesh for the domain.
    :param _sigma_xx0_L: The given function for the effective stress.
    :param _M_0_L: The given function for the p-wave modulus.
    :param _phi_f0: The constant leading-order porosity.
    :return: The first-order porosity on the left \\phi_{f,1}^{L}.
    """
    mesh_array, sigma_xx0_L_array = fenics_to_numpy(_mesh, _sigma_xx0_L)
    mesh_array, M_0_L_array = fenics_to_numpy(_mesh, _M_0_L)
    phi_f0_float = phi_f0.values()[0]
    # Calculate phi_f1_L below
    phi_f1_L = (1 - phi_f0_float) * sigma_xx0_L_array / M_0_L_array
    return mesh_array.transpose()[0], phi_f1_L


def get_us_L(_mesh: Mesh, _phi_f1: Function, _x_c_t: float, _phi_f0: Constant):
    """Calculates the leading order displacement given phi_f1 and phi_f0 according
    to the below formula:

    \\diffp{u_{s,0}}{\\xi_{L}} = \\frac{\\phi_{f,1}x_{c}}{1 - \\phi_{f,0}}

    This function uses scipy's cumtrapz function to approximate the indefinite
    integral. We also have the fixed boundary condition that u_s0(x=1) = 0,
    and so we will adjust the solution after integrating to ensure that u_s0
    is at 0 when \\xi_L is 1.

    :param _mesh: The mesh for the domain.
    :param _phi_f1: The given function for first-order porosity.
    :param _x_c_t: The moving boundary at current time. This comes from the change of variables.
    :param _phi_f0: The constant leading-order porosity.
    :return: The leading-order displacement u_{s,0}.
    """
    mesh_array, phi_f1_array = fenics_to_numpy(_mesh, _phi_f1)
    mesh_dx = mesh_array[1] - mesh_array[0]
    phi_f0_float = phi_f0.values()[0]
    u_s0_shifted = _x_c_t * si.cumtrapz(phi_f1_array, dx=mesh_dx, initial=0) / (1 - phi_f0_float)
    # The below ensures u_s0 = 0 at x = 1
    u_s0 = u_s0_shifted - u_s0_shifted[-1]
    return mesh_array.transpose()[0], u_s0


def get_q(_u_s0: np.array, _sigma_xx0: np.array, _x_c: np.array, _x_c_dot: np.array):
    """Calculates the volume flux, given u_s0 and sigma_xx0. Note that this should be
    constant in space, and so we will average over space (the value is mostly constant).
    Furthermore, due to time derivatives, we will calculate the value of q for all
    times.

    q_{0}(t) = \\diffp{u_{s,0}}{\\tau} -
    \\xi_{L}\\frac{\\dot{x}_{c}}{x_{c}}\\diffp{u_{s,0}}{\\xi_{L}} -
    \\frac{1}{x_{c}}\\diffp{\\sigma_{xx,0}'}{\\xi_{L}}

    :param _u_s0: The leading-order solid displacement.
    :param _sigma_xx0: The leading-order Terzaghi stress.
    :param _x_c: The moving boundary.
    :param _x_c_dot: The time derivative of the moving boundary.
    :return: An array of values q_0(t).
    """
    dxi = 1 / N_x
    dus0_dt = np.gradient(u_s0_xi_array, axis=1) / delta_t
    dsigma_xx0_dxi = np.gradient(sigma_xx0_xi_array, axis=0) / dxi
    dus0_dxi = np.gradient(u_s0_xi_array, axis=0) / dxi

    xarr = np.linspace(0, 1, N_x + 1)
    q = []
    for t in range(sigma_xx0_xi_array.shape[1]):
        q.append(np.mean(dus0_dt[:, t] - dsigma_xx0_dxi[:, t] / x_c[t] -
                         xarr * x_c_dot[t] / x_c[t] * dus0_dxi[:, t]))
    return q


def get_vs_R(_mesh: Mesh, _phi_f1: Function, _x_c_t: float, _phi_f0: Constant,
             _qt: float, _D_phi: Constant):
    """Calculates the solid velocity given phi_f1, phi_f0 and qt according
    to the below formula:

    v_{s,0} = q(t) + \\frac{\\mathcal{D}_{\\phi}}{(1 - \\phi_{f,0})(1 - x_{c}(t))}
    \\diffp{\\phi_{f,1}}{\\xi_{R}}

    :param _mesh: The mesh for the domain.
    :param _phi_f1: The given function for first-order porosity.
    :param _x_c_t: The moving boundary at current time. This comes from the change of variables.
    :param _phi_f0: The constant leading-order porosity.
    :param _qt: The flux at the time t.
    :param _D_phi: The diffusion constant for _phi_f1.
    :return: The leading-order solid velocity v_{s,0}.
    """
    phi_f0_val, D_phi_val = _phi_f0.values()[0], _D_phi.values()[0]
    mesh_array, phi_f1_array = fenics_to_numpy(_mesh, _phi_f1)
    # We multiply by N_x at the end to get the correct form of the spatial gradient
    v_s0 = _qt + D_phi_val / ((1 - phi_f0_val) * (1 - _x_c_t)) * np.gradient(phi_f1_array) * N_x
    return mesh_array.transpose()[0], v_s0


def get_vf_R(_mesh: Mesh, _v_s0: np.array, _phi_f0: Constant, _qt: float):
    """Calculates the fluid velocity given v_s0, phi_f0 and qt according
    to the below formula:

    v_{f,0} = \\frac{1}{\\phi_{f,0}}(q(t) - (1 - \\phi_{f,0})v_{s,0})

    :param _mesh: The mesh for the domain.
    :param _v_s0: The leading-order solid velocity.
    :param _phi_f0: The constant leading-order porosity.
    :param _qt: The flux at the time t.
    :return: The leading-order fluid velocity v_{f,0}.
    """
    phi_f0_val = _phi_f0.values()[0]
    mesh_array, v_s0_array = fenics_to_numpy(_mesh, _v_s0)
    v_f0 = (_qt - (1 - phi_f0_val) * v_s0_array) / phi_f0_val
    return mesh_array.transpose()[0], v_f0


"""
Convert our arrays from (x, t) to (\\xi_L, \\tau) space
"""

c_0_xi_array = x_t_to_xi_L_tau(c_0_array, x_c, i_c)
M_0_xi_array = x_t_to_xi_L_tau(M_0_array, x_c, i_c)
sigma_start = x_t_to_xi_L_tau(sigma_array, x_c, i_c)[:, 0]
# Initial conditions for c_0 and M_0
c_0_L.f.vector().set_local(np.flip(c_0_xi_array[:, 0]))
M_0_L.f.vector().set_local(np.flip(M_0_xi_array[:, 0]))

"""
Define the initial condition
"""

# Expression for the initial conditions
sigma_xx0_L.bind_ic(np.flip(sigma_start))
phi_f1_R.bind_ic(phi_c)

# Define our porosity and displacement
_, phi_array = get_phi_L(mesh, sigma_xx0_L.f, M_0_L.f, phi_f0)
x_coords, u_array = get_us_L(mesh, phi_array, x_c[0], phi_f0)
phi_f1_L = Quantity("$\\phi_{f,1}^{L}$", "Blues", 4, mesh)
u_s0_L = Quantity("$u_{s,0}^{L}$", "Greens", 5, mesh)
phi_f1_L.f = phi_array
u_s0_L.f = u_array

# Define our fluid and solid velocities on the right
qt = 0
_, v_s0_array = get_vs_R(mesh, phi_f1_R.f, x_c[0], phi_f0, qt, D_phi)
_, v_f0_array = get_vf_R(mesh, v_s0_array, phi_f0, qt)
v_s0_R = Quantity("$v_{s,0}^{R}$", "Oranges", 6, mesh)
v_f0_R = Quantity("$v_{f,0}^{R}$", "PuRd", 7, mesh)
v_s0_R.f = v_s0_array
v_f0_R.f = v_f0_array

quantities_L = [c_0_L, M_0_L, sigma_xx0_L, phi_f1_L, u_s0_L]
quantities_R = [phi_f1_R, v_s0_R, v_f0_R]


"""
Set up figure for the overall plot
"""
fig_L, axs_L = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
fig_R, axs_R = plt.subplots(nrows=3, ncols=1, figsize=(4, 30/3), sharex=True)
# fig.subplots_adjust(bottom=0.5)
Quantity.set_axs(quantities_L, axs_L)
Quantity.set_axs(quantities_R, axs_R)
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves and save all our data
"""
saving_L = [True, True, True, True, True]
short_quants_L = ["c", "M", "sigma", "phi", "u"]
file_names_L = [f"xi_L_data/new_q0/{q}_bM_{b_M_num}.csv" for q in short_quants_L]
for file_name in file_names_L:
    pd.DataFrame().to_csv(file_name)

Quantity.plot_quantities(quantities_L, norm, 0.0, saving_L, file_names_L)

saving_R = [True, True, True]
short_quants_R = ["phi", "vs", "vf"]
file_names_R = [f"xi_R_data/new_q0/{q}_bM_{b_M_num}.csv" for q in short_quants_R]
for file_name in file_names_R:
    pd.DataFrame().to_csv(file_name)

Quantity.plot_quantities(quantities_R, norm, 0.0, saving_R, file_names_R)

"""
Loop over time steps and solve in the left region
"""
for n in range(N_time):

    """
    Define the weak form
    """
    # define the time derivatives
    dsigma_xx0_L_dtau = (sigma_xx0_L.f - sigma_xx0_L.f_old) / delta_t

    # Bind c_0 and M_0 to the current timepoint
    c_0_L.f.vector().set_local(np.flip(c_0_xi_array[:, n]))
    M_0_L.f.vector().set_local(np.flip(M_0_xi_array[:, n]))

    # Weak form of sigma_xx0_L includes c_0 and M_0 in the definition
    Fun_sigma_L = ((dsigma_xx0_L_dtau + b_M * c_0_L.f * sigma_xx0_L.f) *
                   sigma_xx0_L.v * dx +
                   x_c_dot[n] / x_c[n] * (sigma_xx0_L.v.dx(0) * x + sigma_xx0_L.v) * sigma_xx0_L.f * dx +
                   (M_0_L.f * sigma_xx0_L.v).dx(0) *
                   sigma_xx0_L.f.dx(0) / (x_c[n]**2) * dx)

    # solve the weak forms
    sigma_xx0_L.solve(Fun_sigma_L)

    # record the porosity and displacement at this time point in the left region
    _, phi_array = get_phi_L(mesh, sigma_xx0_L.f, M_0_L.f, phi_f0)
    x_coords, u_array = get_us_L(mesh, phi_array, x_c[n], phi_f0)
    phi_f1_L.f = phi_array
    u_s0_L.f = u_array

    # plot at the current timepoint
    Quantity.plot_quantities(quantities_L, norm, (n + 1) * delta_t, saving_L, file_names_L)

"""
Recover the volume flux in the left region to be used in the right region
"""
u_s0_xi_array = (pd.read_csv(f"xi_L_data/new_q0/u_bM_{b_M_num}.csv", index_col=0)
                 ).to_numpy()[:, :-1]
sigma_xx0_xi_array = (pd.read_csv(f"xi_L_data/new_q0/sigma_bM_{b_M_num}.csv", index_col=0)
                      ).to_numpy()[:, :-1]
q = get_q(u_s0_xi_array, sigma_xx0_xi_array, x_c, x_c_dot)

"""
Loop over time steps and solve in the right region
"""
for n in range(N_time):

    """
    Define the weak form
    """
    # define the time derivative
    dphi_f1_R_dtau = (phi_f1_R.f - phi_f1_R.f_old) / delta_t

    qt = q[n]
    # Weak form of phi_f1_R includes a Neumann condition
    Fun_phi_R = (phi_f1_R.v * dphi_f1_R_dtau * dx +
                 x_c_dot[n] / (1 - x_c[n]) * (phi_f1_R.v.dx(0) * (1 - x) - phi_f1_R.v) * phi_f1_R.f * dx +
                 D_phi / (1 - x_c[n])**2 * phi_f1_R.v.dx(0) * phi_f1_R.f.dx(0) * dx +
                 (1 - phi_f0) / (1 - x_c[n]) * qt * phi_f1_R.v * ds)

    # solve the weak form
    phi_f1_R.solve(Fun_phi_R)

    # record the solid and fluid velocity at this time point in the right region
    _, v_s0_array = get_vs_R(mesh, phi_f1_R.f, x_c[n], phi_f0, qt, D_phi)
    _, v_f0_array = get_vf_R(mesh, v_s0_array, phi_f0, qt)
    v_s0_R.f = v_s0_array
    v_f0_R.f = v_f0_array

    # plot at the current timepoint
    Quantity.plot_quantities(quantities_R, norm, (n + 1) * delta_t, saving_R, file_names_R)

"""
Left region
"""
# c_0
fig_L.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=c_0_L.cmap),
               orientation='vertical',
               label='$\\tau$', ax=c_0_L.ax)
c_0_L.label_plot(x_label="$\\xi_L$", title="Leading order solute concentration")

# M_0
fig_L.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=M_0_L.cmap),
               orientation='vertical',
               label='$\\tau$', ax=M_0_L.ax)
M_0_L.label_plot(x_label="$\\xi_L$", title="Leading order p-wave modulus")

# sigma_xx,0'
fig_L.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=sigma_xx0_L.cmap),
               orientation='vertical',
               label='$\\tau$', ax=sigma_xx0_L.ax)
sigma_xx0_L.label_plot(x_label="$\\xi_L$", title="Leading order Terzaghi stress")

"""
Plot the solution for the first-order porosity and leading-order displacement
"""
# phi_f,1
fig_L.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1_L.cmap),
               orientation='vertical',
               label='$\\tau$', ax=phi_f1_L.ax)
phi_f1_L.label_plot(x_label="$\\xi_L$", title="First order porosity")

# u_s,0
fig_L.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s0_L.cmap),
               orientation='vertical',
               label='$\\tau$', ax=u_s0_L.ax)
u_s0_L.label_plot(x_label="$\\xi_L$", title="Leading order displacement")

"""
Right region
"""
# phi_f,1 on the right
fig_R.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1_R.cmap),
               orientation='vertical',
               label='$\\tau$', ax=phi_f1_R.ax)
phi_f1_R.label_plot(x_label="$\\xi_R$", title="First order porosity")

# v_s,0 on the right
fig_R.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_s0_R.cmap),
               orientation='vertical',
               label='$\\tau$', ax=v_s0_R.ax)
v_s0_R.label_plot(x_label="$\\xi_R$", title="Leading order solid velocity")

# v_f,0 on the right
fig_R.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_f0_R.cmap),
               orientation='vertical',
               label='$\\tau$', ax=v_f0_R.ax)
v_f0_R.label_plot(x_label="$\\xi_R$", title="Leading order fluid velocity")

# Save figure
fig_L.suptitle(f"FEniCS solution with $b_M={b_M_num}$, left")
fig_L.savefig(f"plots/new_q0/fenics_b_M_{b_M_num}_D_m_{D_m_num}_L.png", bbox_inches="tight")
fig_R.suptitle(f"FEniCS solution with $b_M={b_M_num}$, right")
fig_R.savefig(f"plots/new_q0/fenics_b_M_{b_M_num}_D_m_{D_m_num}_R.png", bbox_inches="tight")
