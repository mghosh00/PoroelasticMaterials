"""
This Python code solves the following system for the fluid velocity within a time
boundary layer, v_{f,0}(\\chi, T)

        \\diffp{v_{f,0}}{T} = \\gamma\\diffp[2]{v_{f,0}}{\\chi},

which will give us then solutions for all other variables in the boundary layer region.

The initial and boundary conditions are:

        v_{f,0}(0, T) = 1, v_{f,0}(\\chi, \\infty) = 1,
        v_{f,0}(\\chi, 0) = 0.

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
import scipy.special as ss
import pandas as pd

from quantity import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True

"""
Define model parameters
"""

# Initial porosity, \\phi_{f,0}
phi_f0 = 0.5

# Degradation parameter
b_M = 1.0

# Péclet number
Pe = 2.0

# Constant (in this regime) p-wave modulus
M_0 = np.exp(-b_M)

# Scaling
epsilon = 0.01
# gamma = M_0 / epsilon
gamma = M_0


"""
Computational parameters
"""

# Size of time step
delta_t = epsilon / 0.1

# Number of time steps
N_time = 100

# Number of mesh points
N_x = 40

"""
Create the mesh
"""

x_mesh = IntervalMesh(N_x, 0, 1)
chi_mesh = IntervalMesh(100, 0, int(1/np.sqrt(epsilon)))
chi_arr = np.linspace(0, 10, 101)

# get the x coodinates
x = x_mesh.coordinates()[:]
# x_arr = np.flip(np.linspace(0, 1, 101))
x_arr = np.linspace(0, 1, 41)


"""
Define the Dirichlet boundary conditions
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


def get_vf_chi(_chi_arr: np.array, T: float, _gamma: float):
    """Finds the analytic solution to v_f0 at T.

    :param _chi_arr: The array of points to evaluate on.
    :param T: The timepoint.
    :param _gamma: Constant gamma.
    :return: v_f0.
    """
    return 1 - ss.erf(_chi_arr / (2 * np.sqrt(_gamma * T)))


def get_vf_x(_x_arr: np.array, T: float, _gamma: float):
    """Finds the analytic solution to v_f0 at T.

    :param _x_arr: The array of points to evaluate on.
    :param T: The timepoint.
    :param _gamma: Constant gamma.
    :return: v_f0.
    """
    return 1 - ss.erf((1 - _x_arr) / (2 * np.sqrt(_gamma * T)))


def get_phif_chi(_chi_arr: np.array, T: float, _gamma: float, _phi_f0: float,
                 _epsilon: float = 1.0):
    """Finds the analytic solution to phi_f,1/2 at T.

    :param _chi_arr: The array of points to evaluate on.
    :param T: The timepoint.
    :param _gamma: Constant gamma.
    :param _phi_f0: Constant initial porosity.
    :param _epsilon: The epsilon parameter. This is for rescaling purposes.
    :return: phi_f,1/2.
    """
    return (_phi_f0 * _chi_arr / _gamma *
            (1 - _phi_f0 - ss.erf(_chi_arr / (2 * np.sqrt(_gamma * T)))) -
            2 * _phi_f0 * np.sqrt(T / (np.pi * _gamma)) *
            np.exp(- _chi_arr**2 / (4 * _gamma * T))) * np.sqrt(_epsilon)


def get_phif_x(_x_arr: np.array, T: float, _gamma: float, _phi_f0: float):
    """Finds the analytic solution to Phi_f,1 at T.

    :param _x_arr: The array of points to evaluate on.
    :param T: The timepoint.
    :param _gamma: Constant gamma.
    :param _phi_f0: Constant initial porosity.
    :param _epsilon: The epsilon parameter. This is for rescaling purposes.
    :return: Phi_f,1.
    """
    return (_phi_f0 * (1 - x_arr) / _gamma *
            (1 - _phi_f0 - ss.erf((1 - x_arr) / (2 * np.sqrt(_gamma * T)))) -
            2 * _phi_f0 * np.sqrt(T / (np.pi * _gamma)) *
            np.exp(- (1 - x_arr)**2 / (4 * _gamma * T)))


def get_sigma_chi(_phi_f1_2: np.array, _phi_f0: float, _gamma: float,
                  _epsilon: float = 1.0):
    """Finds the analytic solution to Sigma_xx0' at T.

    :param _phi_f1_2: The current porosity.
    :param _phi_f0: Constant initial porosity.
    :param _gamma: Constant gamma.
    :param _epsilon: The epsilon parameter. This is for re-scaling purposes.
    :return: sigma_xx0'.
    """
    return _gamma / (1 - _phi_f0) * _phi_f1_2


def get_sigma_x(_phi_f1: np.array, _phi_f0: float, _gamma: float,
                _epsilon: float = 1.0):
    """Finds the analytic solution to Sigma_xx0' at T.

    :param _phi_f1: The current porosity.
    :param _phi_f0: Constant initial porosity.
    :param _gamma: Constant gamma.
    :param _epsilon: The epsilon parameter. This is for re-scaling purposes.
    :return: Sigma_xx0'.
    """
    return _gamma / (1 - _phi_f0) * _phi_f1


def get_u_chi(_chi_arr: np.array, _phi_f1_2: np.array, _phi_f0: float, _epsilon: float = 1.0):
    """Calculates the leading order displacement given phi_f1 and phi_f0 according
    to the below formula:

    \\diffp{u_{s,0}}{x} = \\frac{\\phi_{f,1}}{1 - \\phi_{f,0}}

    This function uses scipy's cumtrapz function to approximate the indefinite
    integral. We also have the fixed boundary condition that u_s0(x=1) = 0,
    and so we will adjust the solution after integrating to ensure that u_s0
    is at 0 when x is 1.

    :param _chi_arr: The mesh for the domain.
    :param _phi_f1_2: The given function for porosity.
    :param _phi_f0: The constant leading-order porosity.
    :return: The leading-order displacement u_{s,0}.
    """
    mesh_dx = _chi_arr[1] - _chi_arr[0]
    u_s0_shifted = - si.cumtrapz(_phi_f1_2 / np.sqrt(_epsilon), dx=mesh_dx, initial=0) / (1 - _phi_f0)
    # The below ensures u_s0 = 0 at \\chi = 0
    u_s0 = u_s0_shifted - u_s0_shifted[0]
    return u_s0


def get_u_x(_x_arr: np.array, _phi_f1: np.array, _phi_f0: float):
    """Calculates the leading order displacement given phi_f1 and phi_f0 according
    to the below formula:

    \\diffp{u_{s,0}}{x} = \\frac{\\phi_{f,1}}{1 - \\phi_{f,0}}

    This function uses scipy's cumtrapz function to approximate the indefinite
    integral. We also have the fixed boundary condition that u_s0(x=1) = 0,
    and so we will adjust the solution after integrating to ensure that u_s0
    is at 0 when x is 1.

    :param _x_arr: The mesh for the domain.
    :param _phi_f1: The given function for porosity.
    :param _phi_f0: The constant leading-order porosity.
    :return: The leading-order displacement u_{s,0}.
    """
    mesh_dx = _x_arr[1] - _x_arr[0]
    u_s0_shifted = si.cumtrapz(_phi_f1, dx=mesh_dx, initial=0) / (1 - _phi_f0)
    # The below ensures u_s0 = 0 at x = 1
    u_s0 = u_s0_shifted - u_s0_shifted[-1]
    return u_s0


"""
Define the initial condition
"""

# Define Quantities
# v_f0 = Quantity("$v_{f,0}$", "PuRd", 0, chi_mesh)
v_f0 = Quantity("$V_{f,0}$", "PuRd", 0, x_mesh)
# sigma_xx0 = Quantity("$\\sqrt{\\epsilon}\\Sigma_{xx,0}'$", "YlOrBr", 1, chi_mesh)
sigma_xx0 = Quantity("$\\Sigma_{xx,0}'$", "YlOrBr", 1, x_mesh)
# phi_f1_2 = Quantity("$\\sqrt{\\epsilon}\\Phi_{f,1/2}$", "Blues", 2, chi_mesh)
phi_f1 = Quantity("$\\Phi_{f,1}$", "Blues", 2, x_mesh)
# u_s0 = Quantity("$u_{s,0}$", "Greens", 3, chi_mesh)
u_s0 = Quantity("$U_{s,0}$", "Greens", 3, x_mesh)


v_f0_array = np.zeros(N_x + 1)
phi_f1_array = np.zeros(N_x + 1)
sigma_array = get_sigma_x(phi_f1_array, phi_f0, gamma, epsilon)
u_array = get_u_x(x_arr, phi_f1_array, phi_f0)

v_f0.f = v_f0_array
phi_f1.f = phi_f1_array
sigma_xx0.f = sigma_array
u_s0.f = u_array

quantities = [v_f0, sigma_xx0, phi_f1, u_s0]

# Set up data saving
saving = [False, False, False, False]
short_quants = ["vf", "sigma", "phi", "us"]
file_names = [f"data_bl/Q_step_new/{q}_bM_{b_M}.csv" for q in short_quants]
for file_name in file_names:
    pd.DataFrame().to_csv(file_name)


"""
Set up figure for the overall plot
"""
fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(4, 50/3), sharex=True)
axs[0].axis('off')
# fig.subplots_adjust(bottom=0.5)
Quantity.set_axs(quantities, axs[1:])
norm = mpl.colors.Normalize(vmin=0.0, vmax=N_time * delta_t)

times = np.linspace(0, N_time * delta_t, N_time + 1)

"""
Plot the initial curves
"""
# Quantity.plot_quantities(quantities, norm, 0.0, saving, file_names,
#                          x_arr=x_arr)
Quantity.plot_quantities(quantities, norm, 0.0, saving, file_names)

"""
Loop over time steps and solve
"""
for n in range(N_time):

    """
    Change the solutions to the updated timepoint.
    """
    v_f0_array = get_vf_x(x_arr, n * delta_t, gamma)
    phi_f1_array = get_phif_x(x_arr, n * delta_t, gamma, phi_f0)
    sigma_array = get_sigma_x(phi_f1_array, phi_f0, gamma, epsilon)
    u_array = get_u_x(x_arr, phi_f1_array, phi_f0)

    v_f0.f = v_f0_array
    phi_f1.f = phi_f1_array
    sigma_xx0.f = sigma_array
    u_s0.f = u_array

    # plot at this timepoint
    # Quantity.plot_quantities(quantities, norm, n * delta_t, saving, file_names,
    #                          x_arr=x_arr)
    Quantity.plot_quantities(quantities, norm, n * delta_t, saving, file_names)


"""
Plot the solutions at the final time step and finalise the plots
"""

# Quantity.plot_quantities(quantities, norm, N_time * delta_t, saving, file_names,
#                          x_arr=x_arr)
Quantity.plot_quantities(quantities, norm, N_time * delta_t, saving, file_names)

# v_f,0
cb_v = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=v_f0.cmap),
                    orientation='vertical',
                    ax=v_f0.ax)
cb_v.set_label(label=r'$T$', fontsize='xx-large')
v_f0.label_plot("", label_size='xx-large')

# sigma_xx,0'
cb_sigma = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=sigma_xx0.cmap),
                        orientation='vertical',
                        ax=sigma_xx0.ax)
cb_sigma.set_label(label=r'$T$', fontsize='xx-large')
sigma_xx0.label_plot("", label_size='xx-large')
sigma_xx0.ax.set_ylim(-2.5, 0.6)

# phi_f,1_2
cb_phi = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=phi_f1.cmap),
                      orientation='vertical',
                      ax=phi_f1.ax)
cb_phi.set_label(label=r'$T$', fontsize='xx-large')
phi_f1.label_plot("", label_size='xx-large')
# phi_f1_2.ax.set_ylim(-0.04, 0.01)
# phi_f1.ax.set_ylim(-0.1, 0.05)
phi_f1.ax.set_ylim(-4.0, 1.0)

# u_s,0
cb_u = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=u_s0.cmap),
                    orientation='vertical',
                    ax=u_s0.ax)
cb_u.set_label(label=r'$T$', fontsize='xx-large')
u_s0.label_plot("", label_size='xx-large',
                x_label=r'$x$')
u_s0.ax.set_ylim(-1, 6)
# u_s0.ax.set_ylim(-1, 15)

# plt.show()
# fig.suptitle(f"FEniCS solution with $b_M={b_M}$ within boundary layer")
fig.savefig(f"plots/Q_step_bl_new/fenics_b_M_{b_M}_Pe_{Pe}_eps_{epsilon}_x.png", bbox_inches="tight")
