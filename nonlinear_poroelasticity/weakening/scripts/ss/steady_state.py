"""
This file contains the functions for calculating the steady state of the system
in the case that E -> E_min.
"""

import numpy as np
import scipy.integrate as si
import scipy.optimize as so

# From steady state analysis, once E -> E_min, we can determine phi_f, a and
# c. We must solve for phi_f and a simultaneously to determine an integration
# constant, B. c can then be recovered.


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

        self.Q_f = params["Q_f"]["Q_f_final"]

        self.c_left = params["bcs"]["c_left"]
        self.sigma_l = params["bcs"]["sigma_left"]
        self.phi_l = self.get_phi_l()

        self.t_phi = (self.mu * self.L ** 2) / (self.k_0 * self.E_star)
        self.t_v = self.L / self.v_star
        self.t_c = self.L ** 2 / self.D_m
        self.factor = (self.t_phi * self.Q_f * self.phi_f0 ** 3) / (self.t_v * self.E_min * (1 - self.phi_f0))

        self.xi = xi

    def get_phi_l(self):
        """Finds the value of phi_f on the left.

        :return: The value of the porosity on the left.
        """
        b = 2 * (1 + self.nu) * (1 - 2 * self.nu) * self.sigma_l / self.E_min + 2 * self.nu
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
        numerator = self.E_min
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

    def solve_iterative(self, phi_f_guess: np.array, max_its: float = 100):
        """If we wish to use the iterative method to solve for phi_f, a and B, then
        we call this method. This has an initial guess for the porosity and then
        uses this to guess a and B and then a new guess for phi_f. This process iterates
        until the phi_f profile ceases to change above a certain tolerance.

        :param phi_f_guess: Initial guess for porosity.
        :param max_its: The maximum number of iterations.
        :return: The steady state for phi_f, a and B as a tuple.
        """
        # Pre-calculated error requirement (and going one order of magnitude lower)
        tol = 1e-7 * self.N_x
        i = 0
        while True:
            print(f"Iteration {i}")
            a_guess = self.calculate_a(phi_f_guess)
            print(f"a_{i}: {a_guess}")
            B_guess = self.calculate_B(a_guess)
            print(f"B_{i}: {B_guess}")
            phi_f_guess_new = self.calculate_phi(a_guess, B_guess)
            sum_squares = ((phi_f_guess_new - phi_f_guess) ** 2).sum()
            print(f"sum_sq_{i}: {sum_squares}")
            phi_f_guess = phi_f_guess_new
            # print(f"phi_f_{i}: {_phi_f_guess}")
            if sum_squares < tol or i == max_its:
                phi_f_ss = phi_f_guess
                a_ss = a_guess
                B_ss = B_guess
                if i == max_its:
                    print(f"Maximum iterations reached ({max_its})")
                break
            i += 1
        return phi_f_ss, a_ss, B_ss

    def solve_analytic(self):
        """Finds the steady state array for phi_f, the value of the left boundary
        and the constant B analytically.

        :return: The steady state for phi_f, a and B as a tuple.
        """
        phi_r = so.fsolve(SteadyState._F_phi_r, self.phi_f0,
                          args=(self.phi_l, self.phi_f0, self.nu, self.factor))[0]
        if self.factor > 0.0:
            a_ss = self.calculate_a_alt(float(phi_r))
            B_ss = self.calculate_B(a_ss)
            phi_f_ss = self.calculate_phi(a_ss, B_ss)
        else:
            B_ss = self.calculate_B(0.0)
            phi_f_ss = self.calculate_phi(0.0, B_ss)
            a_ss = self.calculate_a(phi_f_ss)
        return phi_f_ss, a_ss, B_ss

    def calculate_c(self, phi_f: np.array, a: float):
        """Calculates the steady state value for c given the porosity.

        :param phi_f: The porosity array.
        :param a: The left boundary.
        :return: The predicted steady state profile for the solute concentration.
        """
        dxi = self.xi[1] - self.xi[0]
        exponent_list = []
        for i in range(len(self.xi)):
            integral_i = si.simpson((1 - a) / phi_f[:i + 1], self.xi[:i + 1], dx=dxi)
            exponent_list.append(integral_i)
        exponent = np.array(exponent_list) * self.t_c / self.t_v * self.Q_f
        return self.c_left * np.exp(exponent)

    def calculate_u_s(self, phi_f: np.array, a: float):
        """Calculates the steady state for the displacement given the porosity
        and left boundary.

        :param phi_f: Steady state porosity profile.
        :param a: Left boundary.
        """
        multiplier = (1 - a) / (1 - self.phi_f0)
        dxi = self.xi[1] - self.xi[0]
        integral_list = []
        for i in range(len(self.xi)):
            integral_i = si.simpson(phi_f[:i + 1], self.xi[:i + 1], dx=dxi)
            integral_list.append(integral_i)
        integral_arr = np.array(integral_list)
        return multiplier * (integral_arr - self.phi_f0 * self.xi) + a

    def alternative_B(self):
        """A potential alternative method for calculating B.

        :return: An alternative way of calculating B.
        """
        denominator = 2 * (1 + self.nu) * (1 - 2 * self.nu)
        term2 = 11 / 6 * (1 - self.phi_f0) ** 2
        term3 = - 3 / 2 * (1 - 2 * self.nu)
        return self.factor + (term2 + term3) / denominator
