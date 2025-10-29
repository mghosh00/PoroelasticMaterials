"""
This Python code solves the following nonlinear, nondimensional general system for the
porosity, Young's modulus and solute concentration

        \\frac{t_{\\phi}}{[t]}\\frac{D\\phi_{f}}{Dt} = \\phi_{f}\\frac{\\p}{\\p x}
        \\left[(1 - \\phi_{f})k(\\phi_{f})\\frac{\\p}{\\p x}\\left(E\\g(\\phi_{f})\\right)\\right],
        \\frac{t_{E}}{[t]}\\frac{D^{s}E}{Dt} = -c\\left(E - E_{\\mathrm{min}}\\right),
        \\frac{t_{c}}{[t]}\\phi_{f}\\frac{D^{f}c}{Dt} = \\frac{\\p}{\\p x}\\left(
        \\phi_{f}\\frac{\\p c}{\\p x}\\right),

where the operators D, D^{f} and D^{s} are the material derivatives for the averaged velocity,
fluid velocity and solid velocity respectively. These operators are dependent on E, k, g
(both given functions of the porosity) and v(t), which is the phase-averaged velocity.

The initial conditions are at t = 0:

        \\phi_{f} = \\phi_{f,0} (= const.), E = E_{0}(x), c = c_{0}(x),

with boundary conditions (on a domain [a(t), 1] with left moving boundary):

        v_s = \\frac{t_{v_{s}}}{[t]}\\dot{a}(t) at x = a(t), v_s = 0 at x = 1,
        c = \\frac{1}{t_{c}}\\frac{\\p c}{\\p x} - \\frac{1}{t_{v_{f}}}(c - c_{-\\infty})v_{f} = 0 at x = a(t),
        \\frac{1}{t_{c}}\\frac{\\p c}{\\p x} = 0 at x = 1,

The moving boundary can be determined by the following implicit relation:

        a(t) = \\phi_{f,0} - \\int_{a(t)}^{1}\\phi_{f}(x, t)dx,

given a known profile for \\phi_{f} at the previous timestep (in the numerical scheme).
We will also change coordinates onto a fixed domain (see details below).

The timescales are:

t_{\\phi} = \\frac{\\mu L^{2}}{k_{0}E^{*}},
t_{v_{i}} = \\frac{L}{v_{i}^{*}}, (v_{i} = v, v_{f} or v_{s}),
t_{E} = \\frac{1}{\\beta_{E}c^{*}},
t_{c} = \\frac{L^{2}}{\\mathcal{D}_{m}}.

If we wish to have a nonlinear timestep, we will set our real time t = t(\\tau), where
the function t(\\tau) depends on some parameter \\tau and is defined within the .json
parameter files. The array of \\tau values will be defined on an array of length
N_time with constant spacing delta_tau.
"""

import os

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from nonlinear_poroelasticity.weakening.scripts import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

# Define a class for the left and right boundaries; these functions
# just needs to return the value true when eta is close to
# the boundary 0 or 1
class Left(SubDomain):
    def inside(self, eta, on_boundary):
        return near(eta[0], 0) and on_boundary


class Right(SubDomain):
    def inside(self, eta, on_boundary):
        return near(eta[0], 1) and on_boundary


class SimilaritySolution:
    """A class to run the finite-element method simulation given a list of input parameters.
    """

    def __init__(self, _params: dict, _middle_path: str, _saving: list[bool],
                 _num_quants: int, _plot_xi: bool = True):
        """Initialiser method

        :param _params: Dictionary of all simulation parameters.
        :param _middle_path: Location of plots and data.
        :param _saving: Whether we save data or not.
        :param _num_quants: The number of quantities to be solved for.
        :param _plot_xi: Whether we plot arrays in terms of xi or in terms of eta.
        """

        """
        Computational parameters
        """
        self.params = _params

        # Simulation number (if it exists)
        self.sim_id = 0 if not "sim_id" in self.params else self.params["sim_id"]

        # Number of mesh points
        self.N_eta = self.params["comp"]["N_eta"]

        # Length of domain in eta space
        self.L_eta = self.params["phys"]["L_eta"]

        # Whether we plot log or linear (boolean)
        self.plot_log = True if "plot_log" in self.params["comp"] else False

        self.saving = _saving

        self.num_quants = _num_quants

        self.plot_coord = "xi" if _plot_xi else "eta"
        self.plot_coord_latex = "$\\xi$" if _plot_xi else "$\\eta$"

        """
        Define model parameters
        """

        # Length of domain, L
        L = Constant(self.params["phys"]["L"])

        # Initial porosity, \\phi_{f,0}
        self.phi_f0 = Constant(self.params["ics"]["phi_f"])

        # Weakening parameter, \\beta_{E}
        beta_E = Constant(self.params["phys"]["beta_E"])

        # Poisson ratio and viscosity
        self.nu = Constant(self.params["phys"]["nu"])
        mu = Constant(self.params["phys"]["mu"])

        # Permeability scale
        k_0 = Constant(self.params["scales"]["k"])

        # Solute concentration, Young's modulus and velocity scales
        c_star = Constant(self.params["scales"]["c"])
        E_star = Constant(self.params["scales"]["E"])
        v_star = Constant(self.params["scales"]["v"])

        # Timescales (only parameters other than nu and phi_f0 in the equations)
        t_phi = (mu * L ** 2) / (k_0 * E_star)
        if "Delta p" in self.params["bcs"]:
            t_p = (mu * L ** 2) / (k_0 * self.params["bcs"]["Delta p"] * E_star)
            self.t_v = t_p
        else:
            self.t_v = L / v_star
        t_E = 1 / (beta_E * c_star)
        self.epsilon_E = t_phi / t_E

        self.sigma_l = Constant(self.params["bcs"]["sigma_left"])
        self.gamma = Constant(self.params["bcs"]["Delta p"])

        self.x = Expression('x[0]', degree=1)

        self.phi_f0_num, self.nu_num = self.nums(self.phi_f0, self.nu)
        self.epsilon_E_num, self.gamma_num = self.nums(self.epsilon_E, self.gamma)
        self.sigma_l_num = self.nums(self.sigma_l)[0]

        # Creating file paths
        self.data_path = f"resources/{_middle_path}/data"
        self.plot_path = f"resources/{_middle_path}/plots"

        # Run the initial methods
        self.create_mesh()
        self.initialise_quantities()
        self.assign_boundary_conditions()

    @staticmethod
    def nums(*constants: Constant):
        return tuple([constant(0.0) for constant in constants])

    def create_mesh(self):
        """
        Create the mesh and function space.
        """

        self.mesh = IntervalMesh(self.N_eta, 0, 1)

        # get the eta coodinates
        self.eta = SpatialCoordinate(self.mesh)[0]
        self.eta_arr = np.linspace(0, 1, self.N_eta + 1)

        # Set up function space
        P1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        P0 = FiniteElement("R", self.mesh.ufl_cell(), 0)

        # For vars phi_f, u_s, v, a
        element = MixedElement([P1, P1, P0, P0])
        self.V = FunctionSpace(self.mesh, element)

    def fenics_to_numpy(self, f: Function):
        """Converts a FEniCS function to numpy

        :param f: The function
        :return: The numpy arrays for the coordinates and function
        """
        # If numpy arrays are passed, just return them back
        mesh_array = (self.mesh if isinstance(self.mesh, np.ndarray)
                      else np.array(self.mesh.coordinates()))
        f_array = (f if isinstance(f, np.ndarray)
                   else f.compute_vertex_values(self.mesh))
        return mesh_array, f_array

    def initialise_quantities(self):
        """
        Define the solutions of the problem as quantities and set up the initial conditions.
        """

        f = Quantity("$f$", "Blues", 0, self.mesh)
        h = Quantity("$h$", "Greens", 1, self.mesh)
        self.quantities = {"f": f, "h": h}
        q_names_plotting = list(self.quantities.keys())
        q_names_solving = q_names_plotting + ["v", "a"]
        # Set up the functions from the joint space
        test_functions = TestFunctions(self.V)
        self.test_fn_dict = {q_names_solving[i]: test_functions[i] for i in range(self.num_quants)}

        self.w = Function(self.V)
        functions = split(self.w)
        self.fn_dict = {q_names_solving[i]: functions[i] for i in range(self.num_quants)}

        for i in range(len(q_names_solving) - 2):
            q_name = q_names_solving[i]
            self.quantities[q_name].set_sym_functions(self.fn_dict[q_name], self.test_fn_dict[q_name],
                                                      self.fn_dict[q_name])

    def compute_k(self, _f):
        """Computes k as a function of the porosity.

        :param _f: The effective porosity.
        :return: The effective permeability.
        """
        phi_f = self.phi_f0_num - _f
        numerator = (1 - self.phi_f0) ** 2 * (phi_f ** 3)
        denominator = pow(self.phi_f0, 3) * (1 - phi_f) ** 2
        return numerator / denominator

    def compute_g(self, _f):
        """Computes g as a function of the porosity.

        :param _f: The effective porosity.
        :return: The effective stress.
        """
        phi_f = self.phi_f0_num - _f
        term1 = (1 - self.phi_f0_num) / (1 - phi_f)
        term2 = - 2 * self.nu_num
        term3 = - (1 - 2 * self.nu_num) * (1 - phi_f) / (1 - self.phi_f0_num)
        denominator = 2 * (1 + self.nu_num) * (1 - 2 * self.nu_num)
        return (term1 + term2 + term3) / denominator

    def compute_dg_df(self, _f):
        """Computes the derivative of g as a function of the porosity.

        :param _f: The effective porosity.
        :return: The effective stress.
        """
        phi_f = self.phi_f0_num - _f
        term1 = (1 - self.phi_f0_num) / (1 - phi_f) ** 2
        term2 = 1 - 2 * self.nu_num / (1 - self.phi_f0_num)
        denominator = 2 * (1 + self.nu_num) * (1 - 2 * self.nu_num)
        return - (term1 + term2) / denominator

    def get_f_bc(self, _sigma_xx):
        """Finds the value of f at a point given sigma.

        :param _sigma_xx: The value of sigma' at a point x in the domain.
        :return: The value of the effective porosity at x.
        """
        b = 2 * (1 + self.nu_num) * (1 - 2 * self.nu_num) * _sigma_xx + 2 * self.nu_num
        discriminant = b ** 2 + 4 * (1 - 2 * self.nu_num)
        factor = (1 - self.phi_f0_num) / (2 * (1 - 2 * self.nu_num))
        return self.phi_f0_num - 1 + factor * (discriminant ** (1 / 2) - b)

    def assign_boundary_conditions(self):
        """Assign the boundary conditions of the simulation to each function.
        """
        self.markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.markers.set_all(0)

        # Define the boundary conditions at the left and right
        left = Left()
        left.mark(self.markers, 1)
        right = Right()
        right.mark(self.markers, 2)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.markers)
        bcs = []

        bc_left_f = DirichletBC(self.V.sub(0), self.get_f_bc(self.sigma_l_num - self.gamma_num), self.markers, 1)
        bc_right_f = DirichletBC(self.V.sub(0), self.get_f_bc(self.sigma_l_num), self.markers, 2)

        bc_left_h = DirichletBC(self.V.sub(1), 0.0, self.markers, 1)

        bcs.append(bc_left_f)
        bcs.append(bc_right_f)
        bcs.append(bc_left_h)

        self.bcs = bcs

    def get_sigma_from_E_g(self, _E, _g):
        """Computes the Terzaghi stress as a function of the Young's modulus
        and the porosity.

        :param _E: The Young's modulus.
        :param _g: The effective stress (function of porosity).
        :return: The Terzaghi stress.
        """
        _, E_arr = self.fenics_to_numpy(_E)
        _, g_arr = self.fenics_to_numpy(_g)
        return E_arr * g_arr

    def prepare_figure(self, short_quants, nrows, ncols):
        """Set up figure for the overall plot
        """
        self.short_quants = short_quants
        if self.plot_coord == "xi":
            self.phi_f = Quantity("$\\phi_f$", "Blues", 0, self.mesh)
            self.u_s = Quantity("$u_s$", "Greens", 1, self.mesh)
            self.plotting_quants = [self.phi_f, self.u_s]
            self.short_quants = ["phi_f", "u_s"]
        else:
            self.plotting_quants = [self.quantities[short_quants[i]] for i in range(len(short_quants))]
        self.fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, (10 * nrows)/3), sharex=True)
        plt.subplots_adjust(wspace=1.0 * (ncols - 1))
        if ncols != 1:
            axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
        else:
            axs_list = [axs[i] for i in range(nrows)]
        Quantity.set_axs(self.plotting_quants, axs_list[:])

        for quantity in self.plotting_quants:
            quantity.initialise_dataframe(self.eta_arr)

    def create_weak_form(self):
        """Creates the weak form for the simulation. To be overridden by a user.

        :return: The weak form object.
        """
        f, h = self.quantities["f"], self.quantities["h"]
        a, v = self.fn_dict["a"], self.fn_dict["v"]
        v_a, v_v = self.test_fn_dict["a"], self.test_fn_dict["v"]

        # Effective permeability and effective stress
        k = self.compute_k(f.u)
        g = self.compute_g(f.u)
        dg_deta = g.dx(0) / self.L_eta

        """
        Define the weak form
        """

        # Weak form for the phi equation (f)
        Fun_f = (((self.gamma * v / self.L_eta + 1/2 * self.epsilon_E * self.eta) * f.v).dx(0) * f.u * dx -
                   ((f.u + 1 - self.phi_f0) * k * dg_deta) * f.v.dx(0) / self.L_eta * dx)

        # Weak form for the displacement, h
        Fun_h = (h.u.dx(0) / self.L_eta - f.u / (1 - self.phi_f0)) * h.v * dx

        # Weak form for the phase-averaged velocity
        Fun_v = (self.gamma * v - self.epsilon_E * (1 - self.phi_f0) / (1 - self.phi_f0 + f.u) *
                 (h.u - self.eta * h.u.dx(0)) / 2 - k * dg_deta) * v_v * dx

        # Weak form for the moving boundary
        Fun_a = (a - f.u * self.L_eta / (1 - self.phi_f0)) * v_a * dx

        # Combining the weak forms
        Fun = Fun_f + Fun_h + Fun_v + Fun_a
        return Fun

    def solve(self):
        """Solves the problem using the finite element method.
        """
        f, h = self.quantities["f"], self.quantities["h"]

        Fun = self.create_weak_form()
        # Define the Jacobian, problem and solver
        jacobian = derivative(Fun, self.w)

        """
        Solve
        """
        problem = NonlinearVariationalProblem(Fun, self.w, self.bcs, jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-10
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-9

        # Solve
        solver.solve()
        f.f, h.f, v_, a_f = self.w.split(deepcopy=True)
        if self.plot_coord == "xi":
            self.convert_to_xi(f.f, h.f)
        self.a_num = float(a_f(0.0))
        self.v_num = float(v_(0.0))
        print(f"a*: {self.a_num}, v0: {self.v_num}")

        self.norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        Quantity.plot_quantities(self.plotting_quants, self.norm, 1.0, self.saving)

        # Save all relevant quantities
        if any(self.saving) and not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        file_names = [f"{self.data_path}/ess_{q}_{self.plot_coord}.csv" for q in self.short_quants]
        Quantity.write_to_csv(self.plotting_quants, self.saving, file_names)

    def convert_to_xi(self, _f, _h):
        """Converts functions _f and _h to porosity and displacement, respectively.

        :param _f: FEniCS function representing f.
        :param _h: FEniCS function representing h.
        """
        _, f_arr = self.fenics_to_numpy(_f)
        _, h_arr = self.fenics_to_numpy(_h)
        phi_f_arr = self.phi_f0_num - np.flip(f_arr)
        u_s_arr = np.flip(h_arr) / self.L_eta
        self.phi_f.f = phi_f_arr
        self.u_s.f = u_s_arr

    def save_responses(self):
        """Saves the numbers a and v to a small .csv file.
        """
        responses_file = f"{self.data_path}/ess_responses.csv"
        responses_df = pd.DataFrame({"a": [self.a_num / self.L_eta], "v": [self.v_num * self.L_eta]})
        responses_df.to_csv(responses_file)

    def plot_traces(self):
        """Plots time traces of various quantities.
        """

        # Set up the colorbars and label the plots
        Quantity.annotate_plots(self.plotting_quants, self.fig, self.norm, self.plot_coord_latex)

        # Check plot directory exists
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        # Save figure
        self.fig.savefig(f"{self.plot_path}/ess_time_traces_{self.plot_coord}.png", bbox_inches="tight")
