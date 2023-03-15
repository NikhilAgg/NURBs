from dolfinx import fem, mesh, plot
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
from dolfinx.io import XDMFFile
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from dolfinx_utils import *
from helmholtz_x.passive_flame_x import *
from helmholtz_x.eigensolvers_x import pep_solver, eps_solver, fixed_point_iteration_pep
from helmholtz_x.eigenvectors_x import normalize_eigenvector, normalize_unit, normalize_adjoint
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer 
from helmholtz_x.shape_derivatives_x import _shape_gradient_Robin
from helmholtz_x.shape_derivatives import ShapeDerivatives
from helmholtz_x.parameters_utils import rho, h, gaussianFunction
from helmholtz_x.active_flame_x import ActiveFlameNT4
# from helmholtz_x.petsc4py_utils import conjugate_function
import pyvista

def plotp(p, msh, degree, boundary=1, mesh=False, ret=False):
    point = {
        1: [(0, 0, 0), (0.99999, 0, 0)],
        2: [(0, 0, 0), (0, 0.999999, 0)],
        3: [(0, 0.999999, 0), (1, 0.999999, 0)],
        4: [(0.99999999, 0, 0), (0.9999999, 0.99999999, 0)]
    }
    V = fem.FunctionSpace(msh, ("Lagrange", degree))
    plotter = pyvista.Plotter()
    
    topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid, show_edges=True)

    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    u_grid.point_data["p"] = p.x.array.real
    
    u_grid.point_data["pc"] = p.x.array.imag
    u_grid.set_active_scalars("p")
    plotter.add_mesh(u_grid, show_edges=True)
    x_line = np.linspace(0, 1, 100)
    p_sim = u_grid.sample_over_line(*point[boundary], resolution=99).point_data["p"]
    pc_sim = u_grid.sample_over_line(*point[boundary], resolution=99).point_data["pc"]
    # p_act = np.sqrt(2)*np.pi*np.sin(np.pi*x_line)**2
    # p_act = -np.pi*np.sin(np.pi*x_line)
    # plt.plot(x_line, p_act, label="Analytical")
    if not mesh:
        plt.plot(x_line, p_sim)
        # plt.plot(x_line, p_act, label="Analytical")
        plt.legend()
    else:
        plotter.view_xy()
        plotter.show()
    if ret:
        return p_sim + pc_sim*1j

def square(x_in, x_f, a_f):
    func = []
    for x in x_in:
        if x[0] > x_f[0] - a_f/2 and x[0] < x_f[0] - a_f/2 and x[1] > x_f[1] - a_f/2 and x[1] < x_f[1] - a_f/2:
            func.append(1/(a_f**2))
        else:
            func.append(0)
    
    return np.array(func)

def define_parameters(msh, facet_tags, degree, params):
    rho_u = 1.22
    rho_d = 0.85 
    tau = 0.0015
        
    Q_tot = 200.  # [W]
    U_bulk = 0.1  # [m/s]
    FTF_mag =  0.014  # [/]
    eta = FTF_mag * Q_tot / U_bulk

    x_f = params['x_f']
    a_f = params['a_f']
    # x_f = np.array([0.25, 0.5, 0.])  # [m]
    # a_f = 0.05  # [m]

    x_r = params['x_r']
    a_r = params['a_r']
    # x_r = np.array([0.20, 0.5, 0.])  # [m]
    # a_r = 0.025  # [m]

    V = fem.FunctionSpace(msh, ("Lagrange", degree))

    c = fem.Function(V)
    c.interpolate(lambda x: x[0]*0 + 1.0)

    rh = rho(msh, x_f, a_f, rho_d, rho_u, degree=degree)
    # rh = fem.Function(V)
    # rh.interpolate(lambda x: x[0]*0 + rho_u)
    w = gaussianFunction(msh, x_r, a_r, degree=degree)
    w.vector[facet_tags.indices] = np.zeros(len(facet_tags.indices)) 
    # w = create_func(V, lambda x: square(x, x_r, a_r))
    h_var = gaussianFunction(msh, x_f, a_f, degree=degree)
    # h_var = create_func(V, lambda x: square(x, x_f, a_f))
    h_var.vector[facet_tags.indices] = np.zeros(len(facet_tags.indices)) 

    return [w, h_var, rh, tau, Q_tot, U_bulk, eta, c]


def conjugate_function(p, msh, degree):
    V = fem.FunctionSpace(msh, ("Lagrange", degree))
    p_conj = fem.Function(V)
    p_conj.vector[:] = np.conjugate(p.vector[:])

    return p_conj

def normalize_robin_vectors(omega, A, B, C, D, p, p_adj, R, c, msh, degree):
    L_dash = B + (2*omega)*C + D.get_derivative(omega)
    # L_dash = (2*omega)*C
    V = fem.FunctionSpace(msh, ("Lagrange", degree))
    Lp = fem.Function(V)
    L_dash.mult(p.vector, Lp.vector)

    p_conj = conjugate_function(p_adj, msh, degree)
    Lp = conjugate_function(Lp, msh, degree)

    first_term_val = p_conj.vector.dot(Lp.vector)
    print(first_term_val)

    # V = fem.FunctionSpace(msh, ("Lagrange", degree))
    # Lp = fem.Function(V)
    # B.mult(p.vector, Lp.vector)
    # Lp.vector[:] = Lp.vector[:] + (2*omega)*p.vector[:]

    # first_term = fem.form(ufl.inner(Lp, p_adj)*ufl.dx)
    # first_term_val = fem.assemble_scalar(first_term)
    # print(first_term_val)


    Z = (1+R)/(1-R)

    second_term = fem.form((1j*c/Z) * ufl.inner(p, p_adj)*ufl.ds)
    second_term_val = fem.assemble_scalar(second_term)

    norm_const = first_term_val + second_term_val
    print(second_term_val)

    p_adj.vector[:] = p_adj.vector[:]/np.conjugate(norm_const)
    p_conj = conjugate_function(p_adj, msh, degree)

    print((p_conj.vector.dot(Lp.vector) + fem.assemble_scalar(fem.form((1j*c/Z) * ufl.inner(p, p_adj)*ufl.ds))))
    print(norm_const)
    return [p, p_adj]

def find_eigenvalue(epsilon, degree, N, R, params):
    
    c_const = 1 #np.sqrt(5)
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [[0, -epsilon], [1, 1]], [N, N])
    
    
    

    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], -epsilon)),
              (4, lambda x: np.isclose(x[1], 1))]
    
    ds, facet_tags = tag_boundaries(msh, boundaries=boundaries, return_tags=True)
    w, h, rho, tau, Q_tot, U_bulk, eta, c = define_parameters(msh, facet_tags, degree, params)

    boundary_conditions = {4: {'Robin': R},
                        3: {'Robin': R},
                        2: {'Robin': R},
                        1: {'Robin': R}}

    matrices = PassiveFlame(msh, facet_tags, boundary_conditions, c , degree = degree)

    matrices.assemble_A()
    matrices.assemble_B()
    matrices.assemble_C()

    A = matrices.A
    B = matrices.B
    C = matrices.C

    D = ActiveFlameNT4(msh, [], w, h, rho, Q_tot, U_bulk, eta, tau, 
                    degree=degree)

    # i=0
    # omega = 0
    # while abs(omega.real - 9.9) > 0.05:
    #     target =  np.pi*c_const/(1 + 0.01*i)
    #     E = pep_solver(A, B, C, target**2, nev = 2)
    #     omega, p = normalize_eigenvector(msh, E, 0, degree=degree, which='right')
    #     omega_adj, p_adj = normalize_eigenvector(msh, E, 0, degree=degree, which='left')
        # i += 1
    
    target =  np.pi*c_const*0.7

    E = fixed_point_iteration_pep(matrices, D, target**2, nev=2, i=0, print_results= False)
    omega, p = normalize_eigenvector(msh, E, 0, degree=degree, which='right')
    omega_adj, p_adj = normalize_eigenvector(msh, E, 0, degree=degree, which='left')

    assert np.isclose(omega, omega_adj)

    p = normalize_magnitude(p, imaginary=True)
    p_adj = normalize_magnitude(p_adj, imaginary=True)
    # p_adj = normalize_adjoint(omega, p, p_adj, matrices)

    normalize_robin_vectors(omega, A, B, C, D, p, p_adj, R, c, msh, degree)
    # p_adj = normalize_adjoint(omega, p, p_adj, matrices)
    
    return [omega, p, p_adj, msh, ds, c, facet_tags]

def find_shapegrad_dirichlet(geometry, omega, p, p_adj, ds, c, degree):
    # dp_dn = ufl.Dn(p)
    # G = _shape_gradient_Robin(geometry, c, omega, p, conjugate_function(p_adj, geometry.mesh, degree=degree), 3)
    
    # kappa = geometry.get_curvature_field(3)
    # p_adj = conjugate_function(p_adj, geometry.mesh, degree) 
    # G = - p_adj * (kappa*(c**2)+ c*ufl.Dn(c))*ufl.Dn(p) + ufl.div(p_adj*(c**2)*ufl.grad(p)) - 2*ufl.Dn(p_adj)*(c**2)*ufl.Dn(p)

    shape_der = ShapeDerivatives(geometry, [], p, p_adj, c)
    shape_der.curvature = 0
    G = shape_der.get_Robin()
    C = 1
    dw = fem.assemble_scalar(fem.form(C* G*ds(3)))

    return dw

class Geometry:
    def __init__(self, msh, facet_tags):
        self.mesh = msh
        self.facet_tags = facet_tags
    
    def get_curvature_field(self, index):
        return 0

degree = 2
N = 100
R = -0.9999999999
params = {}
params['x_f'] = np.array([0.25, 0.5, 0])
params['x_r'] = np.array([0.2, 0.5, 0])
params['a_f'] = 0.035
params['a_r'] = 0.025
iters = 5
omega, p, p_adj, msh, ds, c, facet_tags = find_eigenvalue(0, degree, N, R, params)
geometry = Geometry(msh, facet_tags)
dw = find_shapegrad_dirichlet(geometry, omega, p, p_adj, ds, c, degree)
print(dw)
# dw = -2.222073297359943+0.012216679542910025j
# dw = -2.195815977+0.0120058966j #R=-0.975-0.05j
# dw = -2.221435524+0.3397187249j #R=-0.5
# dw = -2.221441469 #R=-1

x_points = [0]
y_points = [0]
epsilon = 0
for i in range(1, iters):
    epsilon += 0.001
    omega_new = find_eigenvalue(epsilon, degree, N, R, params)[0]
    Delta_w_FD = omega_new - omega
    x_points.append(epsilon**2)
    y_points.append(abs(Delta_w_FD - dw*epsilon))

print(x_points)
print(y_points)
# plt.plot(x_points, y_points)
# plt.xlabel('$\epsilon^2$')
# plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
# plt.show()
print(f"THIS IS DW: {dw}")

# x_test = np.linspace(0.1, 0.8, 15)
# a_test = np.linspace(0.01, 0.07, 15)
# R_test = np.linspace(-0.99999999, 0.999999999, 20)
# Final = []
# final_points = []
# for R in R_test:
#     for x_f in x_test:
#         for y_f in x_test:
#             for x_r in x_test:
#                 for y_r in x_test:
#                     for a_f in a_test:
#                         for a_r in a_test:
#                             params = {}
#                             params['x_f'] = np.array([x_f, y_f, 0])
#                             params['x_r'] = np.array([x_r, y_r, 0])
#                             params['a_f'] = a_f
#                             params['a_r'] = a_r
#                             iters = 20
#                             omega, p, p_adj, msh, ds, c, facet_tags = find_eigenvalue(0, degree, N, R, params)
#                             geometry = Geometry(msh, facet_tags)
#                             dw = find_shapegrad_dirichlet(geometry, omega, p, p_adj, ds, c, degree)
#                             print(dw)
#                             # dw = -2.222073297359943+0.012216679542910025j
#                             # dw = -2.195815977+0.0120058966j #R=-0.975-0.05j
#                             # dw = -2.221435524+0.3397187249j #R=-0.5
#                             # dw = -2.221441469 #R=-1

#                             x_points = [0]
#                             y_points = [0]
#                             epsilon = 0
#                             for i in range(1, iters):
#                                 epsilon += 0.005
#                                 omega_new = find_eigenvalue(epsilon, degree, N, R, params)[0]
#                                 Delta_w_FD = omega_new - omega
#                                 x_points.append(epsilon**2)
#                                 y_points.append(abs(Delta_w_FD - dw*epsilon))

#                             print(x_points)
#                             print(y_points)
#                             # plt.plot(x_points, y_points)
#                             # plt.xlabel('$\epsilon^2$')
#                             # plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
#                             # plt.show()
#                             print(f"THIS IS DW: {dw}")

#                             save = [R, x_f, y_f, x_r, y_r, a_f, a_r]
#                             Final.append(save)
#                             points = [x_points, y_points]
#                             final_points.extend(points)

# np.savetxt("tests.csv", Final, delimiter=",")
# np.savetxt("test_coords.csv", final_points, delimiter=",")