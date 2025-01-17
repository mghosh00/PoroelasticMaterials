"""
Here we solve for the deflection of an elastic membrane over the unit disk
"""

# imports
import matplotlib.pyplot as plt
import numpy as np

from fenics import *
from mshr import *

# We create a circle mesh
# Define centre and radius of circle below
domain = Circle(Point(0, 0), 1)
# Second parameter is the mesh resolution
mesh = generate_mesh(domain, 64)
V = FunctionSpace(mesh, 'P', 1)

# Defining a load term
beta = 8
R_0 = 0.6
p = Expression("4 * exp(-pow(beta, 2) * (pow(x[0], 2) + pow(x[1] - R_0, 2)))",
               degree=1, beta=beta, R_0=R_0)

# Defining the variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v)) * dx
L = p * v * dx

# Setting boundary condition


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, 0, boundary)

# Note that we first defined w as a TrialFunction (unknown) but then redefined it
# to be a Function representing the solution - the computed finite element
w = Function(V)
solve(a == L, w, bc)

# We wish to visualise the load, p, along with the deflection. Hence, we will turn
# the FORMULA (Expression) into a FINITE ELEMENT FUNCTION (Function) So, interpolate
# p onto the grid, destroying the previous definition
p = interpolate(p, V)

# Then we can plot w and p
plot(w, title='Deflection')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Deflection of an elastic membrane")
plt.savefig("plots/deflection.png")
plt.clf()
plot(p, title='Load')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Load of an elastic membrane")
plt.savefig("plots/load.png")
plt.clf()

# Plotting both along line x = 0
tol = 0.001 # avoid hitting points outside the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = [(0, y_) for y_ in y] # 2D points
w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])
plt.plot(y, 50*w_line, 'k', linewidth=2) # magnify w
plt.plot(y, p_line, 'b--', linewidth=2)
plt.grid(True)
plt.xlabel('$y$')
plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
plt.savefig('plots/both_curves.png')
