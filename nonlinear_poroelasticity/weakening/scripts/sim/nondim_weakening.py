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
import sys

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json

from nonlinear_poroelasticity.weakening.scripts import Quantity
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

# Define a class for the left and right boundaries; these functions
# just needs to return the value true when xi is close to
# the boundary 0 or 1
class Left(SubDomain):
    def inside(self, xi, on_boundary):
        return near(xi[0], 0) and on_boundary


class Right(SubDomain):
    def inside(self, xi, on_boundary):
        return near(xi[0], 1) and on_boundary


class Simulation:
    """A class to run the finite-element method simulation given a list of input parameters.
    """

    def __init__(self, _params: dict, _middle_path: str, _fixed_domain: bool,
                 _num_quants: int, _saving: list[bool], _num_lines: int):
        """Initialiser method

        :param _params: Dictionary of all simulation parameters.
        :param _middle_path: Location of plots and data.
        :param _fixed_domain: Whether we plot on a domain fixed-in-space or not.
        :param _num_quants: The number of output quantities.
        :param _saving: Whether we save data or not.
        :param _num_lines: The number of lines to plot.
        """

        """
        Computational parameters
        """
        self.params = _params

        # Simulation number (if it exists)
        self.sim_id = 0 if not "sim_id" in self.params else self.params["sim_id"]

        # Size of time step (if the time array has a linear spacing)
        self.delta_tau = self.params["comp"]["delta_tau"]

        # Number of time steps
        self.N_time = self.params["comp"]["N_time"]

        # Number of mesh points
        self.N_x = self.params["comp"]["N_x"]

        # Number of quantities in the solver
        self.num_quants = _num_quants

        # Whether we plot log time or linear time (boolean)
        self.log_time = True if "log_time" in self.params["comp"] else False
        self.fixed_domain = _fixed_domain
        self.plot_coord = "x" if _fixed_domain else "xi"
        self.plot_coord_tex = "$x$" if _fixed_domain else "$\\xi$"

        self.plotting_freq = int(self.N_time / _num_lines)
        self.saving = _saving

        """
        Define model parameters
        """

        # Timescale, [t]
        self.t_sc = Constant(self.params["scales"]["t"])

        # Length of domain, L
        L = Constant(self.params["phys"]["L"])

        # Initial porosity, \\phi_{f,0}
        self.phi_f0 = Constant(self.params["ics"]["phi_f"])

        # Degradation parameter
        beta_E = Constant(self.params["phys"]["beta_E"])

        # Diffusive parameter for the solute concentration
        D_m = Constant(self.params["phys"]["D_m"])

        # Minimum (nondimensional) value of E (can be thought of as
        # fraction of original E)
        self.E_min = Constant(self.params["phys"]["E_min"])

        # Poisson ratio and viscosity
        self.nu = Constant(self.params["phys"]["nu"])
        mu = Constant(self.params["phys"]["mu"])

        # Permeability scale
        k_0 = Constant(self.params["scales"]["k"])

        # Solute concentration, Young's modulus and velocity scales
        c_star = Constant(self.params["scales"]["c"])
        E_star = Constant(self.params["scales"]["E"])
        v_star = Constant(self.params["scales"]["v"])
        v_f_star = Constant(self.params["scales"]["v_f"])
        v_s_star = Constant(self.params["scales"]["v_s"])

        # Timescales (only parameters other than nu and phi_f0 in the equations)
        self.t_phi = (mu * L ** 2) / (k_0 * E_star)
        if "Delta p" in self.params["bcs"]:
            t_p = (mu * L ** 2) / (k_0 * self.params["bcs"]["Delta p"] * E_star)
            self.t_v = t_p
        else:
            self.t_v = L / v_star
        self.t_v_f = self.t_v
        self.t_v_s = self.t_v
        self.t_E = 1 / (beta_E * c_star)
        self.t_c = L ** 2 / D_m

        # c_plus, c_minus = 0, 0
        # if "c_plus" in self.params["bcs"]:
        #     c_plus = Constant(self.params["bcs"]["c_plus"])
        # if "c_minus" in self.params["bcs"]:
        #     c_minus = Constant(self.params["bcs"]["c_minus"])

        self.sigma_l = Constant(self.params["bcs"]["sigma_left"])
        self.Delta_p = Constant(self.params["bcs"]["Delta p"])

        self.x = Expression('x[0]', degree=1)

        self.t_sc_num, self.phi_f0_num, self.nu_num = self.nums(self.t_sc, self.phi_f0, self.nu)
        self.t_v_num, self.t_phi_num, self.t_E_num, self.t_c_num = self.nums(self.t_v, self.t_phi, self.t_E, self.t_c)
        self.sigma_l_num, self.Delta_p_num = self.nums(self.sigma_l, self.Delta_p)

        # Setting up the moving boundary
        self.a_list = [self.params["ics"]["a"]]

        # Creating file paths
        self.data_path = f"resources/{_middle_path}/data"
        self.plot_path = f"resources/{_middle_path}/plots"

        # Run the initial methods
        self.create_mesh()
        self.initialise_timesteps()
        self.initialise_quantities()
        self.assign_boundary_conditions()

    @staticmethod
    def nums(*constants: Constant):
        return tuple([constant(0.0) for constant in constants])

    def create_mesh(self):
        """
        Create the mesh and function space.
        """

        self.mesh = IntervalMesh(self.N_x, 0, 1)

        # get the xi coodinates
        self.xi = SpatialCoordinate(self.mesh)[0]
        self.xi_arr = np.linspace(0, 1, self.N_x + 1)

        # Set up function space
        P1 = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        P0 = FiniteElement("R", self.mesh.ufl_cell(), 0)

        # For vars phi_f, E, c, sigma, u_s, p_f, v, a
        element = MixedElement([P1, P1, P1, P1, P1, P1, P0, P0])
        self.V = FunctionSpace(self.mesh, element)

    def initialise_timesteps(self):
        """Create various quantities representing the time-dependent arrays.
        """
        t_tau = self.params["comp"]["t(tau)"] if "t(tau)" in self.params["comp"] else "tau"
        self.t = Expression(t_tau, degree=1, tau=0.0,
                            delta_tau=self.delta_tau, N_time=self.N_time, domain=self.mesh)
        self.t_next = Expression(t_tau, degree=1, tau=self.delta_tau,
                                 delta_tau=self.delta_tau, N_time=self.N_time, domain=self.mesh)
        self.delta_t = self.t_next - self.t
        self.t_final = float(Expression(t_tau, degree=1, tau=self.N_time * self.delta_tau,
                                        delta_tau=self.delta_tau, N_time=self.N_time, domain=self.mesh)(0.0))

# For calculating the initial value of v (from asymptotic early-time analysis)
# D_phi = (1 - nu_num) / (1 + nu_num) / (1 - 2 * nu_num) * t_sc_num / t_v_num
# v_0 = 1 / np.sqrt(np.pi * D_phi * float(t(0.0))) if early_time_soln else self.params["ics"]["v"]

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

    def xi_t_to_x_t(self, _a: float, *_quantities: Quantity):
        """Change the quantity from (xi, t) coordinates to (x, t) where
        \\xi = 1 - \\frac{1 - x}{1 - a(t)}. We also fit onto the new mesh, which
        will involve some interpolation.

        :param _a: The moving boundary a(t).
        :param _quantities: A tuple of quantities in (xi, t) coordinates (np array).
        :return: The new tuple of arrays in (x, t) coordinates.
        """
        f_part_list = []
        for quantity in _quantities:
            _, _f = self.fenics_to_numpy(quantity.f)
            # How many points there are in the (x, t) domain
            N_part = self.N_x - int(_a * self.N_x)
            # Shrink region to [0, 1] (using transformation) and interpolate onto
            # only the left part of the grid
            f_part = np.interp(np.linspace(0, 1, N_part + 1),
                               np.linspace(0, 1, self.N_x + 1),
                               _f)
            # Fill the left of array with NaNs if domain has been compressed
            if self.N_x - N_part >= 0:
                f_part = np.concatenate([np.full(self.N_x - N_part, np.nan), f_part])
                quantity.mesh_fixed = np.linspace(0, 1, self.N_x + 1)
            # Else, if domain has expanded, we must change the mesh
            else:
                dx = 1 / self.N_x
                N_neg = N_part - self.N_x
                mesh_fixed = np.linspace(- N_neg * dx, 1, self.N_x + N_neg + 1)
                # Update the fixed mesh of the quantity
                quantity.mesh_fixed = mesh_fixed

            f_part_list.append(f_part)

        return tuple(f_part_list)

    def initialise_quantities(self):
        """
        Define the solutions of the problem as quantities and set up the initial conditions.
        """

        phi_f = Quantity("$\\phi_{f}$", "Blues", 0, self.mesh)
        E = Quantity("$E$", "Purples", 1, self.mesh)
        c = Quantity("$c$", "Reds", 2, self.mesh)
        sigma = Quantity("$\\sigma_{xx}'$", "Greys", 3, self.mesh)
        u_s = Quantity("$u_s$", "Greens", 4, self.mesh)
        p_f = Quantity("$p_f$", "Oranges", 5, self.mesh)
        # v = Quantity("$v$", "RdPu", 6, self.mesh)
        # This quantity is just for plotting purposes
        v_s_ = Quantity("$v_{s}$", "YlOrBr", 6, self.mesh)
        self.quantities = {"phi_f": phi_f, "E": E, "c": c, "sigma": sigma, "u_s": u_s, "p_f": p_f, "v_s": v_s_}
        q_names_plotting = list(self.quantities.keys())
        q_names_solving = q_names_plotting[:-1] + ["v", "a"]
        # Set up the functions from the joint space
        test_functions = TestFunctions(self.V)
        self.test_fn_dict = {q_names_solving[i]: test_functions[i] for i in range(self.num_quants)}

        w_0 = self.create_initial_conditions()
        self.w_old = project(w_0, self.V)

        self.w = Function(self.V)
        functions = split(self.w)
        old_functions = split(self.w_old)
        self.fn_dict = {q_names_solving[i]: functions[i] for i in range(self.num_quants)}
        self.old_fn_dict = {q_names_solving[i]: old_functions[i] for i in range(self.num_quants)}
        f_tuple = self.w_old.split(deepcopy=True)
        self.a_f = f_tuple[-1]
        self.v_ = f_tuple[-2]

        for i in range(len(q_names_solving) - 2):
            q_name = q_names_solving[i]
            self.quantities[q_name].set_sym_functions(self.fn_dict[q_name], self.test_fn_dict[q_name],
                                                      self.old_fn_dict[q_name])

            # Associate functions with "f" values (used in plotting/recording)
            self.quantities[q_name].f = f_tuple[i]
        # We don't know the initial array v_s
        v_s_.f = np.full(self.N_x + 1, np.nan)

    def create_initial_conditions(self):
        """Sets up the initial conditions of the simulation.

        :return: A FEniCS function containing the initial conditions.
        """
        # c_ic = 'c_minus + (c_plus - c_minus) * x[0]'
        c_ic = '0.0'
        # phi_ic = ('phi_f0 - (1 - phi_f0) * gamma / D_phi * '
        #           '(1 - erf((1 - x[0]) / (2 * sqrt(D_phi * t1))))')
        # D_phi = (1 - nu_num) / (1 - 2 * nu_num) / (1 + nu_num)
        # t1 = float(t_next(0.0)) * t_E_num / t_phi_num
        # v0 = 1 / np.sqrt(np.pi * D_phi * t1)
        # a0 = 2 * t_phi_num / t_v_num * np.sqrt(t1 / (np.pi * D_phi))
        # print(t1, v0, a0)
        ics = ("phi_f0", self.params["ics"]["E"], c_ic, "0.0",
               self.params["ics"]["u_s"], "0.0", "v_0", "a_0")
        return Expression(ics, degree=1, phi_f0=self.phi_f0, v_0=0.0, a_0=self.a_list[0], 
                          E_min=self.E_min, gamma=self.t_phi_num/self.t_v_num, nu=self.nu)

    def compute_k(self, _phi_f):
        """Computes k as a function of the porosity.

        :param _phi_f: The porosity.
        :return: The effective permeability.
        """
        numerator = (1 - self.phi_f0) ** 2 * (_phi_f ** 3)
        denominator = pow(self.phi_f0, 3) * (1 - _phi_f) ** 2
        return numerator / denominator

    def compute_g(self, _phi_f):
        """Computes g as a function of the porosity.

        :param _phi_f: The porosity.
        :return: The effective stress.
        """
        term1 = (1 - self.phi_f0_num) / (1 - _phi_f)
        term2 = - 2 * self.nu_num
        term3 = - (1 - 2 * self.nu_num) * (1 - _phi_f) / (1 - self.phi_f0_num)
        denominator = 2 * (1 + self.nu_num) * (1 - 2 * self.nu_num)
        return (term1 + term2 + term3) / denominator

    def compute_dg_dphi(self, _phi_f):
        """Computes the derivative of g as a function of the porosity.

        :param _phi_f: The porosity.
        :return: The effective stress.
        """
        term1 = (1 - self.phi_f0_num) / (1 - _phi_f) ** 2
        term2 = 1 - 2 * self.nu_num / (1 - self.phi_f0_num)
        denominator = 2 * (1 + self.nu_num) * (1 - 2 * self.nu_num)
        return (term1 + term2) / denominator

    def get_phi_bc(self, _sigma_xx, _E, _left=True):
        """Finds the value of phi_f at a point given sigma and E.

        :param _sigma_xx: The value of sigma' at a point x in the domain.
        :param _E: The full Young's modulus profile.
        :param _left: Whether we are on the left or right of the domain.
        :return: The value of the porosity at x.
        """
        _, E_arr = self.fenics_to_numpy(_E)
        E_point = float(E_arr[0]) if _left else float(E_arr[-1])
        b = 2 * (1 + self.nu_num) * (1 - 2 * self.nu_num) * _sigma_xx / E_point + 2 * self.nu_num
        discriminant = b ** 2 + 4 * (1 - 2 * self.nu_num)
        factor = (1 - self.phi_f0_num) / (2 * (1 - 2 * self.nu_num))
        return 1 - factor * (discriminant ** (1 / 2) - b)

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

        E = self.quantities["E"]
        bc_left_phi = DirichletBC(self.V.sub(0), self.get_phi_bc(self.sigma_l_num, E.f), self.markers, 1)

        bcs.append(bc_left_phi)
        # Imposed fluid flux or pressure drop

        if "Q_f" in self.params and "Delta p" not in self.params["bcs"]:
            self.Q_f = Expression(self.params["Q_f"]["expr"],
                                  degree=1, t=self.t, delta_tau=self.delta_tau, N_time=self.N_time, domain=self.mesh)
        elif "Delta p" in self.params["bcs"] and "Q_f" not in self.params:
            bc_right_phi = DirichletBC(self.V.sub(0), 
                                       self.get_phi_bc(self.sigma_l_num - self.Delta_p_num, E.f, _left=False),
                                       self.markers, 2)
            bcs.append(bc_right_phi)
            bc_right_sigma = DirichletBC(self.V.sub(3), self.sigma_l_num - self.Delta_p_num, self.markers, 2)
            bcs.append(bc_right_sigma)
            bc_left_pf = DirichletBC(self.V.sub(5), self.Delta_p, self.markers, 1)
            # bcs.append(bc_left_pf)
        else:
            print("Need exactly one of Q_f and Delta p prescribed, exiting...")
            sys.exit()
        
        if "c_left" in self.params["bcs"]:
            bc_left_c = DirichletBC(self.V.sub(2), self.params["bcs"]["c_left"], self.markers, 1)
            bcs.append(bc_left_c)
        if "c_right" in self.params["bcs"]:
            bc_right_c = DirichletBC(self.V.sub(2), self.params["bcs"]["c_right"], self.markers, 2)
            bcs.append(bc_right_c)
        if "sigma_left" in self.params["bcs"]:
            bc_left_sigma = DirichletBC(self.V.sub(3), self.sigma_l, self.markers, 1)
            bcs.append(bc_left_sigma)
        if "u_s_right" in self.params["bcs"]:
            bc_right_us = DirichletBC(self.V.sub(4), self.params["bcs"]["u_s_right"], self.markers, 2)
            bcs.append(bc_right_us)
        if "p_f_right" in self.params["bcs"]:
            bc_right_pf = DirichletBC(self.V.sub(5), self.params["bcs"]["p_f_right"], self.markers, 2)
            bcs.append(bc_right_pf)
        self.bcs = bcs

    def get_vs_from_E_phi(self, _phi_f, _E, _delta_t):
        _, Q_f_arr = self.fenics_to_numpy(self.Q_f)
        _, phi_f_arr = self.fenics_to_numpy(_phi_f)
        _, E_arr = self.fenics_to_numpy(_E)
        k_arr = self.compute_k(phi_f_arr)
        g_arr = self.compute_g(phi_f_arr)
        dg_dphi_arr = self.compute_dg_dphi(phi_f_arr)
        da_dt_val = (self.a_list[-1] - self.a_list[-2]) / _delta_t
        print(da_dt_val)
        _a = self.a_list[-1]
        prod = (E_arr * dg_dphi_arr * np.gradient(phi_f_arr, self.xi_arr) +
                g_arr * np.gradient(E_arr, self.xi_arr))
        _v_s = (Q_f_arr / self.t_v + k_arr * prod
                / ((1 - _a) * self.t_phi)) * self.t_v_s
        return _v_s

    def get_vs_from_u_phi(self, _phi_f, _u_s_new, _u_s_old, _delta_t):
        _, phi_f_arr = self.fenics_to_numpy(_phi_f)
        _, u_s_new_arr = self.fenics_to_numpy(_u_s_new)
        _, u_s_old_arr = self.fenics_to_numpy(_u_s_old)
        dus_dt_arr = (u_s_new_arr - u_s_old_arr) / _delta_t
        da_dt_val = (self.a_list[-1] - self.a_list[-2]) / _delta_t
        print(da_dt_val)
        return (1 / (1 - phi_f_arr) *
                ((1 - self.phi_f0_num) * dus_dt_arr - (1 - self.xi_arr) * da_dt_val * (phi_f_arr - self.phi_f0_num))
                * self.t_v_num / self.t_sc_num)

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
        self.plotting_quants = [self.quantities[short_quants[i]] for i in range(len(short_quants))]
        self.fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, (10 * nrows)/3), sharex=True)
        plt.subplots_adjust(wspace=1.0 * (ncols - 1))
        if ncols != 1:
            axs_list = [axs[i][j] for j in range(ncols) for i in range(nrows)]
        else:
            axs_list = [axs[i] for i in range(nrows)]
        Quantity.set_axs(self.plotting_quants, axs_list[:])
        t_colorbar_init = 1e-12 if self.log_time else 0.0
        if self.log_time:
            self.norm = mpl.colors.LogNorm(vmin=t_colorbar_init, vmax=self.t_final)
        else:
            self.norm = mpl.colors.Normalize(vmin=0.0, vmax=self.t_final)

        for quantity in self.plotting_quants:
            quantity.initialise_dataframe(self.xi_arr)

        Quantity.plot_quantities(self.plotting_quants, self.norm, t_colorbar_init, self.saving)

    def create_weak_form(self):
        """Creates the weak form for the simulation. To be overridden by a user.

        :return: The weak form object.
        """
        phi_f, E, c, sigma, u_s, p_f = (self.quantities["phi_f"], self.quantities["E"],
                                        self.quantities["c"], self.quantities["sigma"],
                                        self.quantities["u_s"], self.quantities["p_f"])
        a, v = self.fn_dict["a"], self.fn_dict["v"]
        a_old, v_old = self.old_fn_dict["a"], self.old_fn_dict["v"]
        v_a, v_v = self.test_fn_dict["a"], self.test_fn_dict["v"]
        print(self.t(0.0), self.t_next(0.0))
        # define the time derivatives
        dphi_dt = (phi_f.u - phi_f.u_old) / self.delta_t
        dE_dt = (E.u - E.u_old) / self.delta_t
        dc_dt = (c.u - c.u_old) / self.delta_t
        da_dt = (a - a_old) / self.delta_t
        dus_dt = (u_s.u - u_s.u_old) / self.delta_t

        # Effective permeability and effective stress
        k = (1 - self.phi_f0) ** 2 * (phi_f.u ** 3) / pow(self.phi_f0, 3) / (1 - phi_f.u) ** 2
        k_div_phi = (1 - self.phi_f0) ** 2 * (phi_f.u ** 2) / pow(self.phi_f0, 3) / (1 - phi_f.u) ** 2
        g = (((1 - self.phi_f0) / (1 - phi_f.u) - 2 * self.nu - (1 - 2 * self.nu) * (1 - phi_f.u) / (1 - self.phi_f0))
             / (2 * (1 + self.nu) * (1 - 2 * self.nu)))
        dg_dphi = ((1 - self.phi_f0) /
                   (1 - phi_f.u) ** 2 + (1 - 2 * self.nu) / (1 - self.phi_f0)) / (2 * (1 + self.nu) * (1 - 2 * self.nu))

        dEg_dxi = (E.u * g).dx(0)

        # Find intermediate expressions for the solid and fluid velocities
        # Below are two different expressions that we need for the solid velocity (they
        # are equivalent definitions)
        self._Q_f = Expression("val", degree=1, val=self.v_(0.0), domain=self.mesh)
        # v_s = t_v_s * (v / t_v + k * p_f.u.dx(0) / ((1 - a) * t_phi))
        _v_s = (self.t_v_s / (1 - phi_f.u) *
                ((1 - self.phi_f0) * dus_dt / self.t_sc - (1 - self.xi) * da_dt * (phi_f.u - self.phi_f0) / self.t_sc))
        # v_f = t_v_f * (v / t_v - (1 - phi_f.u) * k_div_phi * dEg_dxi / ((1 - a) * t_phi))
        # _v_f = t_v_f * (_v_s / t_v_s - k_div_phi * dEg_dxi / ((1 - a) * t_phi))
        _v = self.t_v * (_v_s / self.t_v_s - k * dEg_dxi / ((1 - a) * self.t_phi))
        phi_f_v_f = (v - (1 - phi_f.u) * _v_s)
        # __v = (phi_f.u * _v_f + (1 - phi_f.u) * _v_s)
        # _v = Q_f

        """
        Define the weak form
        """

        # Weak form for the phi equation
        Fun_phi = ((dphi_dt - da_dt * phi_f.u / (1 - a)) * phi_f.v / self.t_sc * dx +
                   ((1 / (1 - a))**2 * (1 - phi_f.u) * k * dEg_dxi / self.t_phi -
                    (1 / (1 - a)) * phi_f.u * (v / self.t_v - (1 - self.xi) * da_dt / self.t_sc)) * phi_f.v.dx(0) * dx +
                   (1 / (1 - a)) * (v / self.t_v) * phi_f.v * self.ds(2) -
                   (1 / (1 - a)) * (v / self.t_v - da_dt / self.t_sc) * phi_f.v * self.ds(1))
        # Fun_phi = ((dphi_dt - da_dt * phi_f.u / (1 - a)) * phi_f.v / t_sc * dx +
        #            ((1 / (1 - a))**2 * phi_f.u * k_e.u * dEg_dx / t_phi -
        #             (1 / (1 - a)) * phi_f.u * (Q_f / t_v_f + xi * da_dt / t_sc)) * phi_f.v.dx(0) * dx +
        #            (Q_f / t_v_f + da_dt / t_sc) * phi_f.v / (1 - a) * ds(2))
        #            (Q_f / t_v_f) * phi_f.v / (1 - a) * ds(1))
        # Fun_phi = ((dphi_dt - da_dt * (1 - xi) / (1 - a) * phi_f.u.dx(0)) / t_sc * phi_f.v * dx +
        #            (phi_f.u * _v_s).dx(0) / (1 - a) / t_v_s * phi_f.v * dx +
        #            k * p_f.u.dx(0) / ((1 - a) ** 2 * t_phi) * phi_f.v.dx(0) * dx)

        # Weak form for the E equation
        Fun_E = (dE_dt / self.t_sc + c.u * (E.u - self.E_min) / self.t_E
                 + (_v_s / self.t_v_s - (1 - self.xi) * da_dt / self.t_sc) / (1 - a) * E.u.dx(0)) * E.v * dx

        # Weak form for the c equation
        # Fun_c = ((phi_f.u * dc_dt + dphi_dt * c.u -
        #           da_dt * c.u * phi_f.u / (1 - a)) / t_sc * c.v * dx +
        #          phi_f.u / (1 - a) *
        #          (c.u.dx(0) / ((1 - a) * t_c) -
        #           (_v_f / t_v_f - (1 - xi) * da_dt / t_sc) * c.u) * c.v.dx(0) * dx +
        #          c.u * phi_f.u * _v_f / (1 - a) / t_v_f * c.v * ds(2))
        # Weak form for the c equation with new boundary conditions
        # Fun_c = (((phi_f.u * dc_dt + dphi_dt * c.u -
        #           da_dt * c.u * phi_f.u / (1 - a)) / t_sc * c.v +
        #          1/ (1 - a) *
        #          (phi_f.u * c.u.dx(0) / ((1 - a) * t_c) -
        #           (phi_f_v_f / t_v_f - phi_f.u * (1 - xi) * da_dt / t_sc) * c.u) * c.v.dx(0)) * dx +
        #          c.u * v / (1 - a) / t_v_f * c.v * ds(2) +
        #          (c.u * da_dt / t_sc - v * c_minus / t_v) / (1 - a) * c.v * ds(1))
        Fun_c = (((phi_f.u * dc_dt + dphi_dt * c.u -
                   da_dt * c.u * phi_f.u / (1 - a)) / self.t_sc * c.v +
                  1/ (1 - a) *
                  (phi_f.u * c.u.dx(0) / ((1 - a) * self.t_c) -
                   (phi_f_v_f / self.t_v_f - phi_f.u * (1 - self.xi) * da_dt / self.t_sc) * c.u) * c.v.dx(0)) * dx +
                 c.u * v / (1 - a) / self.t_v_f * c.v * self.ds(2) +
                 (da_dt / self.t_sc - v / self.t_v) / (1 - a) * c.v * self.ds(1))

        # Weak form for the Terzaghi stress (sort of Lagrange multiplier)
        Fun_sigma = (sigma.u - (E.u * g)) * sigma.v * dx

        # Weak form for the displacement
        Fun_us = ((u_s.u.dx(0) * u_s.v -
                   (phi_f.u - self.phi_f0) * (1 - a) / (1 - self.phi_f0) * u_s.v) * dx)

        # Weak form for the fluid pressure
        Fun_pf = (sigma.u - p_f.u) * p_f.v.dx(0) * dx + (sigma.u - p_f.u) * p_f.v * self.ds(1)

        # Weak form for the phase-averaged velocity
        if "Q_f" in self.params:
            Fun_v = (((self.Q_f - v) / self.t_v) * v_v * dx)
        else:
            Fun_v = (v - _v) * v_v * dx

        # Weak form for the moving boundary
        Fun_a = ((phi_f.u - 1) + (1 - self.phi_f0) / (1 - a)) * v_a * dx

        # Combining the weak forms
        Fun = Fun_phi + Fun_E + Fun_c + Fun_sigma + Fun_us + Fun_pf + Fun_v + Fun_a
        return Fun

    def solve(self):
        """Solves the problem using the finite element method.
        """
        phi_f, E, c, sigma, u_s, p_f, v_s_ = (self.quantities["phi_f"], self.quantities["E"],
                                              self.quantities["c"], self.quantities["sigma"],
                                              self.quantities["u_s"], self.quantities["p_f"], self.quantities["v_s"])
        Fun = self.create_weak_form()
        # Define the Jacobian, problem and solver
        jacobian = derivative(Fun, self.w)

        """
        Loop over time steps and solve
        """
        t_list = [float(self.t(0.0))]
        # Lists of averages to record (Q_f, E_avg, c_avg, phi_f_avg)
        Q_f_list = [self.v_(0.0)]
        avgs_dict = {"phi_f": [phi_f.get_average()], "E": [E.get_average()], "c": [c.get_average()]}
        # c_right_list = [float(c_plus(0.0))]
        phi_r_list = [self.get_phi_bc(self.sigma_l_num - self.Delta_p_num, E.f, _left=False)]
        # integral_v_list = [float(integral_v(0.0))]
        # t.tau += delta_tau
        # t_next.tau += delta_tau
        for n in range(self.N_time):
            self.t_fl = float(self.t_next(0.0))
            problem = NonlinearVariationalProblem(Fun, self.w, self.bcs, jacobian)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters["nonlinear_solver"] = "newton"
            solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-10
            solver.parameters["newton_solver"]["relative_tolerance"] = 1e-9
            print("Time:", np.round(self.t_fl, 3))

            # Solve
            solver.solve()
            phi_f.f, E.f, c.f, sigma.f, u_s_new, p_f.f, v_, a_f = self.w.split(deepcopy=True)
            _, phi_f_arr = self.fenics_to_numpy(phi_f.f)
            self.a_list.append(a_f(0.0))

            self._Q_f.val = v_(0.0)
            Q_f_list.append(float(self._Q_f(0.0)))
            # c_right_list.append(float(fenics_to_numpy(mesh, c.f)[1][-1]))
            # Update some variables
            self.t.tau += self.delta_tau
            self.t_next.tau += self.delta_tau
            t_list.append(float(self.t(0.0)))
            delta_t_fl = t_list[-1] - t_list[-2]

            # Record various averages
            avgs_dict["phi_f"].append(phi_f.get_average())
            avgs_dict["E"].append(E.get_average())
            avgs_dict["c"].append(c.get_average())

            # v_s_.f = get_vs_from_E_phi(mesh, phi_f.f, E.f, a_list, phi_f0_num, nu_num,
            #                            t_v_num, t_v_s_num, t_phi_num, delta_t_fl)
            v_s_.f = self.get_vs_from_u_phi(phi_f.f, u_s_new, u_s.f, delta_t_fl)
            self.w_old.assign(self.w)

            # Change coordinates onto the fixed domain for plotting
            (phi_f.f_fixed, E.f_fixed, c.f_fixed, sigma.f_fixed,
             u_s.f_fixed, p_f.f_fixed, v_s_.f_fixed) = self.xi_t_to_x_t(self.a_list[-1], phi_f, E,
                                                                        c, sigma, u_s, p_f, v_s_)

            # plot at the current timepoint if needed
            if phi_f.f_fixed[-1] < 0.0:
                phi_f.f_fixed[-1] = 0.0
            phi_r = phi_f.f_fixed[-1]
            u_s.f = u_s_new

            phi_l = self.fenics_to_numpy(phi_f.f)[1][0]
            if phi_l < 0.0:
                print("Porosity on the left has reached zero, exiting...")
                break
            bc_left_phi = DirichletBC(self.V.sub(0), self.get_phi_bc(self.sigma_l_num, E.f), self.markers, 1)
            self.bcs[0] = bc_left_phi
            phi_r = self.get_phi_bc(self.sigma_l_num - self.Delta_p_num, E.f, _left=False)
            bc_right_phi = DirichletBC(self.V.sub(0), phi_r, self.markers, 2)
            self.bcs[1] = bc_right_phi

            if phi_r < 0.0:
                print("Porosity on the right has reached zero, exiting...")
                phi_r_list.append(0.0)
                break
            phi_r_list.append(phi_r)
            if (n + 1) % self.plotting_freq == 0:
                Quantity.plot_quantities(self.plotting_quants, self.norm, self.t_fl, self.saving,
                                         fixed_domain=self.fixed_domain)

        # Save all relevant quantities
        if any(self.saving) and not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        file_names = [f"{self.data_path}/_{q}_{self.plot_coord}.csv" for q in self.short_quants]
        Quantity.write_to_csv(self.plotting_quants, self.saving, file_names)

        # Creating np arrays for later
        self.times = np.array(t_list)
        self.Q_f_arr = np.array(Q_f_list)
        self.phi_f_avg_arr = np.array(avgs_dict["phi_f"])
        self.E_avg_arr = np.array(avgs_dict["E"])
        self.c_avg_arr = np.array(avgs_dict["c"])
        self.phi_fr_arr = np.array(phi_r_list)

    def plot_traces(self):
        """Plots time traces of various quantities.
        """

        # Set up the colorbars and label the plots
        Quantity.annotate_plots(self.plotting_quants, self.fig, self.norm, self.plot_coord_tex)

        # Check plot directory exists
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        # Save figure
        self.fig.savefig(f"{self.plot_path}/_time_traces_{self.plot_coord}.png", bbox_inches="tight")

    def plot_responses(self):
        # Create figure for the imposed velocity and left boundary over time
        fig_Q_a, axs_Q_a = plt.subplots(nrows=2, ncols=1, figsize=(8, 20 / 3), sharex=True)
        ax_Q, ax_a = axs_Q_a
        ax_Q.plot(self.times[1:], np.array(self.Q_f_arr[1:]), lw=2,
                  color='forestgreen')
        ax_Q.set_ylabel("$v(t)$")
        ax_Q.set_xscale("log")
        ax_Q.set_yscale("log")
        ax_a.plot(self.times, np.array(self.a_list), lw=2,
                  color='darkgoldenrod')
        ax_a.set_ylabel("$a(t)$")
        ax_a.set_xlabel("$t$")
        fig_Q_a.savefig(f"{self.plot_path}/Q_a.png", bbox_inches="tight")

    def plot_t_tau(self):
        fig_tau, ax_tau = plt.subplots(nrows=1, ncols=1, figsize=(8, 10 / 3))
        ax_tau.plot(np.linspace(0, self.N_time * self.delta_tau, len(self.times)), self.times, lw=2,
                    color='darkviolet')
        ax_tau.set_ylabel("$t$")
        ax_tau.set_xlabel("$\\tau$")
        ax_tau.set_yscale("log")
        fig_tau.savefig(f"{self.plot_path}/tau.png", bbox_inches="tight")

    def plot_averages(self):
        # Create figure for various averages over time
        fig_avgs, ax_all = plt.subplots(figsize=(8, 3))

        # Plot E_avg, c_avg and phi_r over time on the same axis
        ax_all.plot(self.times, self.phi_f_avg_arr, lw=2,
                    color="mediumblue", label="$\\overline{\\phi_{f}}$")
        ax_all.plot(self.times, self.E_avg_arr, lw=2,
                        color="indigo", label="$\\overline{E}$")
        ax_all.plot(self.times, self.c_avg_arr, lw=2,
                        color="red", label="$\\overline{c}$")
        ax_all.set_xlabel("$t$")
        if self.log_time:
            ax_all.set_xscale("log")
        ax_all.legend()
        plt.subplots_adjust(hspace=0.3, wspace=0.4)
        fig_avgs.savefig(f"{self.plot_path}/averages.png", bbox_inches="tight")

    def save_responses_and_averages(self):
        if self.saving[0]:
            # Save various time-dependent variables to a dataframe
            response_df = pd.DataFrame({"Time": self.times, "a": np.array(self.a_list), "v": self.Q_f_arr,
                                        "phi_f_bar": self.phi_f_avg_arr, "E_bar": self.E_avg_arr,
                                        "c_bar": self.c_avg_arr, "phi_r": self.phi_fr_arr})
            response_df.to_csv(f"{self.data_path}/_responses.csv")

    def save_t_crit_vals(self):
        """Saving t_crit values
        """
        t_crit_path = f"{self.data_path}/t_crit.csv"
        if not os.path.exists(t_crit_path):
            t_crit_df = pd.DataFrame(columns=["sim_id", "phi_f0", "E_min", "gamma", "t_E", "t_crit"])
        else:
            t_crit_df = pd.read_csv(f"{self.data_path}/t_crit.csv", index_col=0)
        self.t_crit = self.t_fl
        t_crit_df.loc[len(t_crit_df)] = [self.sim_id, self.phi_f0_num, self.nums(self.E_min)[0],
                                         self.t_phi_num / self.t_v_num, self.t_E_num, self.t_crit]
        t_crit_df.to_csv(f"{self.data_path}/t_crit.csv")
