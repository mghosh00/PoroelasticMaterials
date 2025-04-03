"""A class to represent a physical quantity.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from warnings import simplefilter

from fenics import *

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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
        self.mesh_fixed = None
        self._expression = expression
        self.f = None
        self.f_fixed = None
        self.g = None
        self.create_functions_from_mesh(mesh)
        self.interpolate()
        self.ax = None
        self._bcs = []
        self.df = None

    def create_functions_from_mesh(self, mesh):
        """Provided the mesh is not None, we define the function space, solution,
        test function and solution at previous timestep here, every time the mesh
        is reset. Before we do this, we store the current solution as a numpy array.

        :param mesh: The new mesh
        """
        if mesh is not None:
            if self.f is not None:
                _, self.array = self.fenics_to_numpy(self._mesh, self.f)
            self.V = FunctionSpace(mesh, FiniteElement("Lagrange", interval, 1))
            self.f = Function(self.V)
            self.v = TestFunction(self.V)
            self.g_old = Function(self.V)
            self.g = self.f

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

    def initialise_dataframe(self, x_array: np.array):
        """Initialises the dataframe (storing the time history of the quantity)
        with the x-coordinate. This is to save reading and writing to a .csv
        every timestep.

        :param x_array: The domain.
        """
        self.df = pd.DataFrame({"x": x_array})

    def interpolate(self):
        """If there is a valid expression, interpolate this onto the function space.
        """
        if self._expression is not None:
            self.f = interpolate(self._expression, self.V)
            _, self.array = self.fenics_to_numpy(self._mesh, self.f)

    def bind_ic(self, ic):
        """Binds an initial condition to the function.

        :param ic: The np array, Constant or Expression representing the initial
        condition.
        """
        if isinstance(ic, np.ndarray):
            self.g.vector().set_local(ic)
        else:
            self.g.interpolate(ic)
        self.g_old.assign(self.g)
        _, self.array = self.fenics_to_numpy(self._mesh, self.g)

    def solve(self, weak_form: Function):
        """Solves the quantity one step forward in time for the given weak form.

        :param weak_form: The given expression for the numerical solver.
        """
        # Define the Jacobian, problem and solver
        jacobian = derivative(weak_form, self.g)
        problem = NonlinearVariationalProblem(weak_form, self.g, self._bcs, jacobian)
        solver = NonlinearVariationalSolver(problem)

        # Solve the problem
        solver.solve()

        # Update the old solution
        self.g_old.assign(self.g)

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
             save_data: bool = False, fixed_domain: bool = False,
             label: str = None):
        """Plots the curve at the current timepoint (dictated by the col_val).

        :param norm: A normalising function for the colorscale.
        :param time: A float for the current timepoint.
        :param save_data: Whether we save the data or not.
        :param fixed_domain: Whether we plot on the fixed or transformed domain.
        :param label: Whether the plot should have a label.
        """
        if fixed_domain and self.f_fixed is not None:
            f_array = self.f_fixed
            mesh_array = self.mesh_fixed
        else:
            f = self.f
            mesh_array, f_array = self.fenics_to_numpy(self._mesh, f)
        if save_data:
            diff = len(mesh_array) - len(self.df)
            if diff > 0:
                # In this case, the domain has expanded, so we add new
                # rows of NaNs to the *beginning* of the dataframe.
                nan_rows = np.empty((diff, len(self.df.columns)))
                nan_rows[:] = np.nan
                new_df = pd.DataFrame(nan_rows, columns=self.df.columns)
                df = pd.concat([new_df, self.df], ignore_index=True)
                df['x'] = mesh_array
            self.df[time] = f_array
        if self.ax:
            line = self.ax.plot(mesh_array, f_array, color=self.cmap(norm(time)),
                                label=label)
            return line

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
                        fixed_domain: bool = False):
        """Plots multiple quantities at the given time.

        :param quantities: The list of quantities.
        :param norm: The norm to be applied.
        :param time: The current timepoint.
        :param save_list: Whether we save (choice for each quantity).
        :param fixed_domain: Whether we plot on the fixed domain or the transformed one
        """
        lines = []
        if save_list:
            for i, quantity in enumerate(quantities):
                line = quantity.plot(norm, time, save_list[i],
                                     fixed_domain=fixed_domain)
                lines.append(line)
        else:
            for i, quantity in enumerate(quantities):
                line = quantity.plot(norm, time,
                                     fixed_domain=fixed_domain)
                lines.append(line)
        return lines

    @staticmethod
    def write_to_csv(quantities, saving: list[bool], filenames: list[str]):
        """Writes the df of each quantity to a .csv file.
        """
        for i in range(len(quantities)):
            if saving[i]:
                quantities[i].df.to_csv(filenames[i])

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

    def annotate_panel(self, fig: plt.Figure, norm: mpl.colors.Normalize,
                       xlabel: str, tlabel: str = "$t$", title: str = None,
                       ymin: float = None, ymax: float = None):
        """Sets up the colorbar for an axis and sets up the labels.

        :param fig: The overall figure object.
        :param norm: The Normalize object.
        :param xlabel: The x label.
        :param tlabel: The t label.
        :param title: The optional title of the panel.
        :param ymin: The optional minimum value of the quantity (for all time).
        :param ymax: The optional maximum value of the quantity (for all time).
        """
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=self.cmap),
                     orientation='vertical',
                     label=tlabel, ax=self.ax)
        self.label_plot(x_label=xlabel, title=title)
        if ymin and ymax:
            self.ax.set_ylim(ymin, ymax)

    @staticmethod
    def annotate_plots(quantities, fig, norm, xlabel, tlabel="$t$", titles=None,
                       mins=None, maxes=None):
        """Sets up colorbars and labels for a list of quantities.

        :param quantities: The list of quantities.
        :param fig: The overall figure object.
        :param norm: The Normalize object.
        :param xlabel: The x label.
        :param tlabel: The t label.
        :param titles: The optional titles of the panels.
        :param mins: The optional minimal values of the quantities (for all time).
        :param maxes: The optional maximal values of the quantities (for all time).
        """
        if titles is None:
            titles = [None] * len(quantities)
        if mins is None:
            mins = [None] * len(quantities)
        if maxes is None:
            maxes = [None] * len(quantities)
        for i, quantity in enumerate(quantities):
            quantity.annotate_panel(fig, norm, xlabel, tlabel, titles[i],
                                    mins[i], maxes[i])

    def __str__(self):
        return self._name
