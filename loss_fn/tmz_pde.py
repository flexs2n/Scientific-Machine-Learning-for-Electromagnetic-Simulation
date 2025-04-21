from physicsnemo.sym.eq.pde import PDE
from sympy import Symbol, Function, Number


class TMz_PDE(PDE):
    """TMz Maxwell's PDEs using PhysicsNeMo Sym"""

    name = "TMz_PDE"

    def __init__(self, epsilon_0=8.854187817e-12, mu_0=4e-7 * 3.141592653589793, sigma=0.0, relative_permittivity=1.0):
        # x, y, time
        x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # make functions
        Ez = Function("Ez")(*input_variables)
        Hx = Function("Hx")(*input_variables)
        Hy = Function("Hy")(*input_variables)
        Jz = Function("Jz")(*input_variables)
        epsilon = Function("epsilon")(*input_variables)
        sigma = Function("sigma")(*input_variables)

        Ez_rhs = Function("Ez_rhs")(*input_variables)
        Hx_rhs = Function("Hx_rhs")(*input_variables)
        Hy_rhs = Function("Hy_rhs")(*input_variables)

        # initialize constants
        epsilon_0 = Number(epsilon_0)
        mu_0 = Number(mu_0)
        sigma = Number(sigma)
        relative_permittivity = Number(relative_permittivity)

        # set equations
        self.equations = {}

        # Gradients
        self.equations["Hy_x"] = Hy.diff(x)
        self.equations["Hx_y"] = Hx.diff(y)
        self.equations["Ez_x"] = Ez.diff(x)
        self.equations["Ez_y"] = Ez.diff(y)

        # RHS of TMz equations
        self.equations["Ez_rhs"] = (1 / epsilon) * (self.equations["Hy_x"] - self.equations["Hx_y"] - sigma * Ez - Jz)
        self.equations["Hx_rhs"] = -(1 / mu_0) * self.equations["Ez_y"]
        self.equations["Hy_rhs"] = (1 / mu_0) * self.equations["Ez_x"]

        # PDE residuals
        self.equations["DEz"] = Ez.diff(t) - Ez_rhs
        self.equations["DHx"] = Hx.diff(t) - Hx_rhs
        self.equations["DHy"] = Hy.diff(t) - Hy_rhs