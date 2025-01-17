"""
Poisson equation:

-\\nabla^{2}u(\\mathbf{x}) = f(\\mathbf{x}), \\mathbf{x} \\in \\Omega

u(\\mathbf{x}) = u_{D}(\\mathbf{x}), \\mathbf{x} \\in \\partial D

"""
from fenics import *
import matplotlib.pyplot as plt

# Setting up a square domain
N = 20
mesh = UnitSquareMesh(N, N)

# Setting up our function space
# domain, polynomial functions, triangular elements
V = FunctionSpace(mesh, 'P', 1)

# Setting up the boundary conditions
# u_D is this on the boundary, with degree 2 allowing for quadratic terms
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)


# FEniCS knows where the boundary of the domain is, this is just an explicit way
# of writing out the boundary (commented out below)
def boundary(x, on_boundary):
    return on_boundary
    # Alternative formulation
    # return near(x[0], 0) or near(x[0], 1) or near(x[1], 0) or near(x[1], 1)


bc = DirichletBC(V, u_D, boundary)

# Defining the variational problem on the function space
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)

# Creating the weak formulation
a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Finding the solution
u = Function(V)
solve(a == L, u, bc)

# Plotting solution and mesh
plot(u)
plot(mesh)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Solution to Poisson's equation")
plt.savefig("plots/poisson_soln.png")

# Saving the solution to a file
vtkfile = File("vtk/poisson_soln.pvd")
vtkfile << u
