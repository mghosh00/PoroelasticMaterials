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
"""
import os
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

plot_coord = "zeta"
plot_coord_tex = "$\\zeta$"

"""
Define model parameters
"""

# Length of domain, L
L = Constant(params["phys"]["L"])

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(params["ics"]["phi_f"])

# Poisson ratio and viscosity
nu = Constant(params["phys"]["nu"])
mu = Constant(params["phys"]["mu"])

# Permeability scale
k_0 = Constant(params["scales"]["k"])

# Solute concentration, Young's modulus and velocity scales
E_star = Constant(params["scales"]["E"])
v_star = Constant(params["scales"]["v"])
v = Constant(params["v"]["v_final"])

# Timescales (only parameters other than nu and phi_f0 in the equations)
t_phi = (mu * L ** 2) / (k_0 * E_star)
t_v = L / v_star

# The collective "diffusion coefficient" for the equation
D_phi = ((1 * (1 - phi_f0) * ((1 - phi_f0) ** 2 + 1 - 2 * nu) * t_v) /
         (8 * t_phi * (1 + nu) * (1 - 2 * nu) * phi_f0 ** 3))


def nums(*constants: Constant):
    return tuple([constant(0.0) for constant in constants])


phi_f0_num, nu_num = nums(phi_f0, nu)
t_v_num = nums(t_v)
t_phi_num, D_phi_num = nums(t_phi, D_phi)

zeta_max = 1
h_max = D_phi_num ** (-1/4)

"""
Computational parameters
"""

# Number of mesh points
N_x = params["comp"]["N_x"]

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, zeta_max)
zeta_arr = np.linspace(0, zeta_max, N_x + 1)
zeta = SpatialCoordinate(mesh)[0]

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


"""
Define the solution h
"""

h = Quantity("$h$", "Reds", 0, mesh)


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
# the boundary zeta_max
def right(xi):
    return near(xi[0], zeta_max)


# Define the boundary conditions at the left and right
bc_right_h = DirichletBC(h.V, h_max, right)
h.add_bc(bc_right_h)
# h.add_bc(DirichletBC(h.V, 0, left))

param_file.close()

quantities = [h]


"""
Set up figure for the overall plot
"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 10/3), sharex=True)
h.set_ax(ax)
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

"""
Plot the initial curves and save all our data
"""
saving = [True]

for quantity in quantities:
    quantity.initialise_dataframe(zeta_arr)

"""
Define the weak form
"""

Fun_h = (5 * h.g * h.v * dx + (4 * zeta * h.g + 5 * D_phi * (h.g ** 4).dx(0)) * h.v.dx(0) * dx -
         5 * v * zeta / zeta_max * h.v * ds)
# Fun_h = (5 * h.g * h.v * dx + (4 * zeta * h.g + 20 * D_phi * pow(h.g, 3) * h.g.dx(0)) * h.v.dx(0) * dx -
#          5 * v * zeta / zeta_max * h.v * ds)
# parameters["nonlinear_variational_solver"]["relative_tolerance"] = 1e-7
# parameters["nonlinear_variational_solver"]["absolute_tolerance"] = 1e-7

"""
Solve
"""

# Solve and plot
h.solve(Fun_h)
h.plot(norm, 0.0, True)

# Save all relevant quantities
data_path = f"resources/{trial}/{sub_trial}/data"
if any(saving) and not os.path.isdir(data_path):
    os.makedirs(data_path)
file_names = [f"{data_path}/h.csv"]
Quantity.write_to_csv(quantities, saving, file_names)

"""
Colourbars
"""

h.label_plot(x_label=plot_coord_tex, title="")


# Check plot directory exists
plot_path = f"resources/{trial}/{sub_trial}/plots"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Save figure
fig.savefig(f"{plot_path}/h_sim_soln.png", bbox_inches="tight")
