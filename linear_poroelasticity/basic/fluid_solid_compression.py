"""
This Python code solves the 1D diffusion equation for the first-order
porosity, \\phi_{f,1}

        \\diffp{\\phi_{f,1}}{t} = \\diffp[2]{\\phi_{f,1}}{x}

The boundary conditions are:

        \\phi_{f,1}(0, t) = (1 - \\phi_{f,0})\\sigma'^{*}
        \\phi_{f,1}(1, t) = (1 - \\phi_{f,0})(\\sigma'^{*}-\\Delta p)

The initial condition is

        \\phi_{f,1}(x, 0) = 0

"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

"""
Define model parameters
"""

# Initial porosity, \\phi_{f,0}
phi_f0 = Constant(0.5)

# Compressive stress applied at the left boundary
sigma_prime_star = Constant(-0.5)

# Pressure difference across the domain
Delta_p = Constant(0.5)


"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 40

# Number of mesh points
N_x = 40

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, 1)

# get the x coodinates
x = mesh.coordinates()[:]


"""
Define the elements used to represent the solution
"""

# use piecewise linear functions
P1 = FiniteElement("Lagrange", interval, 1)

# create a function space on the mesh
V = FunctionSpace(mesh, P1)

"""
Define the solution \\phi_{f,1} and its test function v
"""

phi_f1 = Function(V)
v = TestFunction(V)

# also define the solution at the previous time step
phi_f1_old = Function(V)

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
bc_left = DirichletBC(V, (1 - phi_f0) * sigma_prime_star, left)
bc_right = DirichletBC(V, (1 - phi_f0) * (sigma_prime_star - Delta_p), right)

# Create a list for all of the boundary conditions
all_bcs = [bc_left, bc_right]

"""
Define the weak form
"""

# define the time derivative of u
dudt = (phi_f1 - phi_f1_old) / delta_t

# Define the weak form of the diffusion equation
Fun = dudt * v * dx + phi_f1.dx(0) * v.dx(0) * dx

# Compute the Jacobian of the weak form (needed for the solver)
Jac = derivative(Fun, phi_f1)


"""
Define the problem and solver
"""
problem = NonlinearVariationalProblem(Fun, phi_f1, all_bcs, Jac)
solver = NonlinearVariationalSolver(problem)

"""
Define the initial condition
"""

# Expression for the initial condition
phi_f1_ic = Constant(0)

# interpolate the initial condition to the mesh points and store in the solution u
phi_f1.interpolate(phi_f1_ic)

# set u_old equal to the values in u
phi_f1_old.assign(phi_f1)


def plot_curve(_mesh: Mesh, u: Function, alpha: float = 1):
    mesh_array = np.array(_mesh.coordinates())
    u_array = u.compute_vertex_values(_mesh)
    plt.plot(mesh_array, u_array, '-b', alpha=alpha)


"""
Loop over time steps and solve
"""
plot_curve(mesh, phi_f1, alpha=0.5)
for n in range(N_time):

    # solve the weak form
    solver.solve()

    # update u_old with the new solution
    phi_f1_old.assign(phi_f1)
    plot_curve(mesh, phi_f1, alpha=(N_time + n)/(2 * N_time))


"""
Plot the solution at the final time step
"""
plot_curve(mesh, phi_f1)
plt.xlabel("$x$")
plt.ylabel("$\\phi_{f,1}$")
plt.title("First-order porosity over time under mechanical and fluid-driven compression")
plt.show()
plt.savefig("plots/both_sigma_0.5_p_0.5.png", bbox_inches="tight")
