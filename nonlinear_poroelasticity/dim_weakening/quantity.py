"""A class to represent a physical quantity.
"""
import numpy as np
import matplotlib as mpl
import pandas as pd

from fenics import *


class Quantity:
    """Class to represent a physical quantity.
    """

    def __init__(self, name: str, cmap_name: str, pos: int,
                 mesh: Mesh = None, expression: Expression = None):
        """Initializer

        :param name: The name of the quantity.
        :param cmap_name: The colourmap name for the plots.
        :param pos: The position on the plot.
        :param mesh: The current mesh of interest.
        :param expression: An optional expression for the quantity (if it is known).
        """
        self._name = name
        whole_map = mpl.colormaps[cmap_name]
        self.cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{cmap_name}_subset",
                                                                 whole_map(np.linspace(0.3, 1.0, 100)))
        self._pos = pos
        self._mesh = mesh
        self._expression = expression
        self.g = None
        self.f = None
        self.create_functions_from_mesh(mesh)
        self.interpolate()
        self.ax = None
        self._bcs = []

    def create_functions_from_mesh(self, mesh):
        """Provided the mesh is not None, we define the function space, solution,
        test function and solution at previous timestep here, every time the mesh
        is reset. Before we do this, we store the current solution as a numpy array.

        :param mesh: The new mesh
        """
        if mesh is not None:
            if self.f is not None:
                _, self.f_np = self.fenics_to_numpy(self._mesh, self.f)
            self.V = FunctionSpace(mesh, FiniteElement("Lagrange", interval, 1))
            self.f = Function(self.V)
            self.v = TestFunction(self.V)
            self.f_old = Function(self.V)

    def set_sym_functions(self, g: None, v: None, g_old: None):
        """If we wish to set the functions manually, we can set these using the
        above parameters.

        :param g: The function.
        :param v: The test function.
        :param g_old: The old function (at previous timestep).
        """
        if g is not None:
            self.g = g
        if v is not None:
            self.v = v
        if g_old is not None:
            self.g_old = g_old

    def interpolate(self):
        """If there is a valid expression, interpolate this onto the function space.
        """
        if self._expression is not None:
            self.f = interpolate(self._expression, self.V)
            _, self.f_np = self.fenics_to_numpy(self._mesh, self.f)

    def bind_ic(self, ic):
        """Binds an initial condition to the function.

        :param ic: The np array, Constant or Expression representing the initial
        condition.
        """
        if isinstance(ic, np.ndarray):
            self.f.vector().set_local(ic)
        else:
            self.f.interpolate(ic)
        self.f_old.assign(self.f)
        _, self.f_np = self.fenics_to_numpy(self._mesh, self.f)

    def solve(self, weak_form: Function):
        """Solves the quantity one step forward in time for the given weak form.

        :param weak_form: The given expression for the numerical solver.
        """
        # Define the Jacobian, problem and solver
        jacobian = derivative(weak_form, self.f)
        problem = NonlinearVariationalProblem(weak_form, self.f, self._bcs, jacobian)
        solver = NonlinearVariationalSolver(problem)

        # Solve the problem
        solver.solve()

        # Update the old solution
        self.f_old.assign(self.f)

    @staticmethod
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

    def plot(self, norm: mpl.colors.Normalize, time: float,
             save_data: bool = False, file_name: str = ''):
        """Plots the curve at the current timepoint (dictated by the col_val).

        :param norm: A normalising function for the colorscale.
        :param time: A float for the current timepoint.
        :param save_data: Whether we save the data or not.
        :param file_name: The name of our file if we save the data.
        """
        mesh_array, f_array = self.fenics_to_numpy(self._mesh, self.f)
        if self.ax:
            self.ax.plot(mesh_array, f_array, color=self.cmap(norm(time)))
        if save_data:
            df = pd.read_csv(file_name, index_col=0)
            df[time] = f_array
            df.to_csv(file_name)

    def label_plot(self, title, x_label='', y_label='name', label_size=None):
        """Label the plot for the quantity.

        :param x_label: Standard x_label.
        :param title: Title for subplot.
        :param y_label: Optional title (will just be the name of quantity otherwise).
        :param label_size: Text size (optional).
        """
        self.ax.set_xlabel(x_label, fontsize=label_size)
        self.ax.set_ylabel(y_label if y_label != 'name' else self._name, fontsize=label_size)
        self.ax.set_title(title)

    @staticmethod
    def plot_quantities(quantities, norm: mpl.colors.Normalize,
                        time: float, save_list: list[bool] = None,
                        file_names: list[str] = None):
        """Plots multiple quantities at the given time.

        :param quantities: The list of quantities.
        :param norm: The norm to be applied.
        :param time: The current timepoint.
        :param save_list: Whether we save (choice for each quantity).
        :param file_names: The names of our files.
        """
        if save_list:
            for i, quantity in enumerate(quantities):
                quantity.plot(norm, time, save_list[i], file_names[i])
        else:
            for i, quantity in enumerate(quantities):
                quantity.plot(norm, time)

    def add_bc(self, bc):
        """Stores a boundary condition in the bcs list.

        :param bc: The boundary condition.
        """
        self._bcs.append(bc)

    def set_ax(self, ax):
        """Stores the ax from a figure

        :param ax: The new ax.
        """
        self.ax = ax

    @staticmethod
    def set_axs(quantities, axs):
        """Sets axes for a number of quantities.

        :param quantities: A list of Quantities.
        :param axs: A list of Axes.
        """
        for i in range(len(quantities)):
            quantities[i].set_ax(axs[i])
