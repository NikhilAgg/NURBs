from model import Model, param_dict
from shapederivative import *
from dolfinx import fem, mesh
from dolfinx_utils2 import create_func
from mpi4py import MPI
from helmholtz_x.parameters_utils import rho, gaussianFunction
from dolfinx_utils import tag_boundaries
import matplotlib.pyplot as plt

def new_msh(epsilon, N=100):
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0, -epsilon], [1, 1]], [N, N])
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], -epsilon)),
              (4, lambda x: np.isclose(x[1], 1))]
    
    _, facet_tags = tag_boundaries(msh, boundaries=boundaries, return_tags=True)

    return msh, facet_tags

N = 100
degree = 2
cell_type = "Lagrange"
msh, facet_tags = new_msh(0, N)
V = fem.FunctionSpace(msh, (cell_type, degree))

#Parameters
def new_params(ms, U):
    rho_u = 1.22
    rho_d = 0.85 
    tau = 0.0015
    Q_tot = 200.  # [W]
    U_bulk = 0.1  # [m/s]
    FTF_mag =  0.014  # [/]
    eta = FTF_mag * Q_tot / U_bulk

    x_f = np.array([0.25, 0.5, 0.])  # [m]
    a_f = 0.035  # [m]

    x_r = np.array([0.20, 0.5, 0.])  # [m]
    a_r = 0.025  # [m]

    c = fem.Function(U)
    c.interpolate(lambda x: x[0]*0 + 1.0)
    rh = rho(ms, x_f, a_f, rho_d, rho_u, degree=degree)
    w = gaussianFunction(ms, x_r, a_r, degree=degree)
    h = gaussianFunction(ms, x_f, a_f, degree=degree)

    return param_dict(c, w, h, rh, tau, Q_tot, U_bulk, eta, False)

params = new_params(msh, V)

#Boundary Conditions
R = -0.5
bound_type = "Robin"
bound_cond = {4: {bound_type: R},
                    3: {bound_type: R},
                    2: {bound_type: R},
                    1: {bound_type: R}}

#Run
model = Model(msh, facet_tags, degree, bound_cond, params, cell_type)
shapeder = ShapeDerivative(model)

target = (np.pi*0.7)**2
x, y = shapeder.perform_taylor_test(1, 3, new_msh, new_params, 0.01, target, 2, 0, 5)
print(x, y)
plt.plot(x, y)
plt.show()