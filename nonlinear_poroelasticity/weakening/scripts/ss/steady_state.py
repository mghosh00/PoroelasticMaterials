"""
This file contains the functions for calculating the steady state of the system
in the case that E -> E_min.
"""

import numpy as np
import scipy.integrate as si
import scipy.optimize as so


class SteadyState:
    """A class to calculate the steady state of a system.
    """

    def __init__(self, params: dict, xi: np.array):
        """Class initialiser.

        :param params: All the parameters from the simulation, coming from a .json file.
        :param xi: The spatial coordinate array of the problem.
        """
        self.N_x = params["comp"]["N_x"]
        self.phi_f0 = params["ics"]["phi_f"]

        self.L = params["phys"]["L"]
        self.nu = params["phys"]["nu"]
        self.mu = params["phys"]["mu"]
        self.E_min = params["phys"]["E_min"]
        self.D_m = params["phys"]["D_m"]

        self.k_0 = params["scales"]["k"]
        self.E_star = params["scales"]["E"]
        self.v_star = params["scales"]["v"]
        if "c_left" in params["bcs"]:
            self.c_left = params["bcs"]["c_left"]

        # # Whether we close the valve once c reaches its steady state or not
        # self.close_valve = params["close_valve"] if "close_valve" in params else 0
        # # If we close the valve, the steady state value E_min changes, so we apply this change here
        # if self.close_valve:
        #     self.E_final = self.E_min + (1 - self.E_min) / np.e
        # else:
        #     self.E_final = self.E_min

        self.E_final = self.E_min
        self.sigma_l = params["bcs"]["sigma_left"]
        self.phi_l = self.get_phi(self.sigma_l)

        self.t_phi = (self.mu * self.L ** 2) / (self.k_0 * self.E_star)
        self.t_v = self.L / self.v_star
        self.t_c = self.L ** 2 / self.D_m
        if "Q_f" in params:
            self.Q_f = params["Q_f"]["Q_f_final"]
            self.fluid_flux = True
            self.factor = (self.t_phi * self.Q_f * self.phi_f0 ** 3) / (self.t_v * self.E_final * (1 - self.phi_f0))
        else:
            self.Delta_p = params["bcs"]["Delta p"]
            self.fluid_flux = False

        self.xi = xi

    def get_phi(self, sigma):
        """Finds the value of phi_f given a stress sigma.

        :return: The value of the porosity.
        """
        b = 2 * (1 + self.nu) * (1 - 2 * self.nu) * sigma / self.E_final + 2 * self.nu
        discriminant = b ** 2 + 4 * (1 - 2 * self.nu)
        multiplier = (1 - self.phi_f0) / (2 * (1 - 2 * self.nu))
        return 1 - multiplier * (discriminant ** (1 / 2) - b)

    @staticmethod
    def _F(_phi_f: np.array, *args):
        """The relation for the analytic
        solution of the steady state of phi_f.

        :param _phi_f: The porosity array.
        :param args: The remaining arguments.
        :return: The right hand side of the steady state relation.
        """
        _phi_f0 = args[0]
        _nu = args[1]
        _B = args[2]
        _factor = args[3]
        _a = args[4]
        _xi = args[5]
        denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
        factor1 = (1 - _phi_f0) ** 2
        term1 = 1 / (3 * (1 - _phi_f) ** 3) - 3 / (2 * (1 - _phi_f) ** 2) + 3 / (1 - _phi_f) + np.log(1 - _phi_f)
        factor2 = 1 - 2 * _nu
        term2 = 1 / (1 - _phi_f) + 3 * np.log(1 - _phi_f) - 3 * (1 - _phi_f) + 1 / 2 * (1 - _phi_f) ** 2
        _x = _a + (1 - _a) * _xi
        return (factor1 * term1 + factor2 * term2) / denominator + _factor * _x - _B

    @staticmethod
    def _F_phi_r_inner(u: float, _phi_f0: float, _nu: float):
        """An inner expression for determining the overall expression for phi_r.

        :param u: A placeholder for (1 - phi_f) where phi_f is any porosity.
        :return: An inner expression used to further determine phi_r.
        """
        term1 = - 1 / (2 * u ** 2) + 3 / u + 3 * np.log(u) - u
        term2 = np.log(u) - 3 * u + 3 / 2 * u ** 2 - u ** 3 / 3
        return (1 - _phi_f0) ** 2 * term1 + (1 - 2 * _nu) * term2

    @staticmethod
    def _F_phi_r(_phi_r: float, *args):
        """The relation for determining the value of phi on the right hand
        side of the domain (at x = xi = 1).

        :param _phi_r: The value of phi on the RHS.
        :param args: The remaining args.
        :return: The equation for this relation (F_phi_r == 0).
        """
        _phi_l = args[0]
        _phi_f0 = args[1]
        _nu = args[2]
        _factor = args[3]
        denominator = 2 * (1 + _nu) * (1 - 2 * _nu)
        term1 = SteadyState._F_phi_r_inner(1 - _phi_r, _phi_f0, _nu)
        term2 = SteadyState._F_phi_r_inner(1 - _phi_l, _phi_f0, _nu)
        return (term1 - term2) / denominator - (1 - _phi_f0) * _factor

    def find_gamma_crit(self):
        """Finds gamma_crit, the value of gamma (the ratio of the timescales) for which
        phi_r = 0 in the steady state. This will likely not be the gamma used in the
        simulation.
        """
        numerator = self.E_final
        denominator = self.phi_f0 ** 3 * self.Q_f * 2 * (1 + self.nu) * (1 - 2 * self.nu)
        term1 = self._F_phi_r_inner(1, self.phi_f0, self.nu)
        term2 = self._F_phi_r_inner(1 - self.phi_l, self.phi_f0, self.nu)
        return numerator / denominator * (term1 - term2)

    def inner_expr_for_a(self, u: np.array):
        """An inner expression used in calculating the value of the left boundary, a.

        :param u: A placeholder for 1 - phi_f (where phi_f is some porosity).
        """
        term1 = 1 / (3 * u ** 3) - 2 / (u ** 2) + 6 / u + 4 * np.log(u) - u
        term2 = 1 / u + 4 * np.log(u) - 6 * u + 2 * u ** 2 - u ** 3 / 3
        return (1 - self.phi_f0) ** 2 * term1 + (1 - 2 * self.nu) * term2

    def calculate_B(self, a: float):
        """Given the above parameters, finds the value that B must take.

        :param a: Position of the left boundary.
        :return: The guess for B.
        """
        return SteadyState._F(self.phi_l, self.phi_f0, self.nu, 0, self.factor, a, 0)

    def calculate_phi(self, a: float, B: float):
        """Uses scipy.optimize to invert the relation between phi and xi.

        :param a: The guess for the left boundary.
        :param B: The guess for the integration constant, B.
        :return: The array phi_f.
        """
        phi_f_initial = np.array([self.phi_f0] * len(self.xi))
        phi_f_list = []
        for i in range(len(self.xi)):
            phi_f_i = so.fsolve(self._F, phi_f_initial[i],
                                args=(self.phi_f0, self.nu, B, self.factor, a, self.xi[i]))
            phi_f_list.append(phi_f_i[0])
        phi_f = np.array(phi_f_list)
        return phi_f

    def calculate_a(self, phi_f: np.array):
        """Calculates the value of the left boundary given the porosity.

        :param phi_f: The porosity array.
        :return: The integral of a function of porosity.
        """
        dxi = self.xi[1] - self.xi[0]
        integrand = phi_f
        integral = si.simpson(integrand, self.xi, dx=dxi)
        return (self.phi_f0 - integral) / (1 - integral)

    def calculate_a_alt(self, phi_r: float):
        """An alternative way to calculate the value of the left boundary, using
        the left and right values of the porosity and other constants.

        :param phi_r: The right value of the porosity.
        :return: The final value of the left boundary.
        """
        denominator = 2 * (1 + self.nu) * (1 - 2 * self.nu)
        term1 = self.inner_expr_for_a(1 - phi_r)
        term2 = self.inner_expr_for_a(1 - self.phi_l)
        return self.phi_f0 + (term1 - term2) / (self.factor * denominator)

    def solve_analytic(self):
        """Finds the steady state array for phi_f, the value of the left boundary
        and the constant B analytically.

        :return: The steady state for phi_f, a and B as a tuple.
        """
        if self.fluid_flux:
            phi_r = so.fsolve(SteadyState._F_phi_r, self.phi_f0,
                              args=(self.phi_l, self.phi_f0, self.nu, self.factor))[0]
        else:
            phi_r = self.get_phi_r_pressure_drop()
        if np.isnan(phi_r):
            phi_f_ss = np.empty(self.N_x + 1)
            phi_f_ss[:], a_ss, B_ss = np.nan, np.nan, np.nan
        elif self.factor > 0.0:
            a_ss = self.calculate_a_alt(float(phi_r))
            B_ss = self.calculate_B(a_ss)
            phi_f_ss = self.calculate_phi(a_ss, B_ss)
        else:
            B_ss = self.calculate_B(0.0)
            phi_f_ss = self.calculate_phi(0.0, B_ss)
            a_ss = self.calculate_a(phi_f_ss)
        return phi_f_ss, a_ss, B_ss

    def get_phi_r_pressure_drop(self):
        """Finds the value for phi_r with an imposed pressure drop
        and finds the steady state value of v, to be used in factor.

        :return: The steady state for phi_r
        """
        phi_r = self.get_phi(self.sigma_l - self.Delta_p)
        v_ss = self.calculate_v(phi_r)
        self.update_factor(v_ss)
        phi_r = np.nan if phi_r < 0 else phi_r
        return phi_r

    def calculate_v(self, phi_r):
        """Calculates the phase-averaged velocity given the porosity on the right.

        :param phi_r: The porosity on the right.
        :return: The phase-averaged velocity in the steady state.
        """
        multiplier = (self.E_final * self.t_v / (self.phi_f0 ** 3 * self.t_phi)
                      if self.phi_f0 != 0 else 1e32)
        denominator = 2 * (1 + self.nu) * (1 - 2 * self.nu)
        term1 = self._F_phi_r_inner((1 - phi_r), self.phi_f0, self.nu)
        term2 = self._F_phi_r_inner((1 - self.phi_f0), self.phi_f0, self.nu)
        return multiplier / denominator * (term1 - term2)

    def update_factor(self, v):
        """Updates the phase-averaged velocity in factor.

        :param v: New phase-averaged velocity.
        """
        self.Q_f = v
        self.factor = (self.t_phi * self.Q_f * self.phi_f0 ** 3) / (self.t_v * self.E_final * (1 - self.phi_f0))

    def get_alpha_min(self):
        """Calculates the alpha_min parameter, which determines whether the
        steady state is physical or not. If alpha_min > 1, a steady state
        is attainable. Otherwise, it is not.

        :return: alpha_min.
        """
        numerator = self.E_min * self.phi_f0 * (2 * (1 - self.nu) - self.phi_f0)
        denominator = self.Delta_p * 2 * (1 - self.phi_f0) * (1 + self.nu) * (1 - 2 * self.nu)
        alpha_min = numerator / denominator if denominator != 0 else 1e32
        return alpha_min
