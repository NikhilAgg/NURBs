import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

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
from helmholtz_x.eigensolvers_x import eps_solver
from helmholtz_x.eigenvectors_x import normalize_eigenvector, normalize_unit
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer 
from helmholtz_x.petsc4py_utils import vector_matrix_vector, conjugate_function
from geoms import *
from bspline.nurbs_geometry import NURBs2DGeometry
from dolfinx_utils.utils import *


def create_mesh(r, epsilon, lc, degree, typ, bspline_ind, param_ind):
    square = circle(r, 0, lc)
    geom = NURBs2DGeometry([[square]])
    edit_param_2d(geom, typ, bspline_ind, param_ind, epsilon)
    geom.model.remove_physical_groups()
    geom.model.remove()
    geom.generate_mesh()
    geom.add_bspline_groups([[9]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", degree)

    return geom


def create_mesh2(r, epsilon, lc, degree):
    geom = circle2(r, epsilon, lc, show_mesh=False)
    geom.V = fem.FunctionSpace(geom.msh, ("Lagrange", degree))

    return geom


def normalize_robin_vectors(omega, A, C, p, p_adj, c, geom):
    L_dash = (2*omega)*C
    Lp = fem.Function(geom.V)
    L_dash.mult(p.vector, Lp.vector)

    first_term_val = conjugate_function(p_adj).vector.dot(Lp.vector)
    print(first_term_val)

    # first_term = fem.form(ufl.inner(p_adj, Lp)*ufl.dx)
    # first_term_val = fem.assemble_scalar(first_term)

    Z=1
    second_term = fem.form((1j*c/Z) * ufl.inner(p_adj, p)*ufl.ds)
    second_term_val = fem.assemble_scalar(second_term)

    norm_const = first_term_val + second_term_val

    p.vector[:] = p.vector[:]/np.sqrt(norm_const)
    p_adj.vector[:] = p_adj.vector[:]/np.sqrt(norm_const)

    return [p, p_adj]


def find_eigenvalue(r, epsilon, typ, bspline_ind, param_ind):
    degree = 3
    c_const = np.sqrt(1)
    geom = create_mesh(r, epsilon, 1e-1, degree, typ, bspline_ind, param_ind)
    c = fem.Constant(geom.msh, PETSc.ScalarType(c_const))

    boundary_conditions = {9: {'Dirichlet'}}
    ds = ufl.ds
    facet_tags = geom.facet_tags

    matrices = PassiveFlame(geom.msh, facet_tags, boundary_conditions, c , degree = degree)

    matrices.assemble_A()
    matrices.assemble_C()

    A = matrices.A
    C = matrices.C

    target =  c_const * np.pi
    E = eps_solver(A, C, target**2, nev = 2, two_sided=True)
    omega, p = normalize_eigenvector(geom.msh, E, 0, degree=degree, which='right')
    omega_adj, p_adj = normalize_eigenvector(geom.msh, E, 0, degree=degree, which='left')

    p = normalize_magnitude(p)
    p_adj = normalize_magnitude(p_adj)
    
    p, p_adj = normalize_robin_vectors(omega, A, C, p, p_adj, c, geom)

    return [omega, p, p_adj, geom, ds, c]


def find_shapegrad_dirichlet(r, omega, p, p_adj, geom, ds, c, typ, bspline_ind, param_ind, tie, ep_list):
    G = -c**2*ufl.Dn(p)*ufl.Dn(p_adj)
    C = fem.Function(geom.V)
    geom.create_node_to_param_map()
    C = np.dot(geom.get_displacement_field(typ, bspline_ind, [param_ind], flip=True, tie=tie), ep_list)

    dw = fem.assemble_scalar(fem.form(G*C*ds))

    return dw


cache = False
ep_step = 0.001
r = 1.
param_ind = 0
typ = "weight"
bspline_ind = (0, 0)

if typ == "control point":
    tie = True
    ep_list = np.array([1., 0.])
else:
    tie = True
    ep_list = 1


omega, p, p_adj, geom, ds, c = find_eigenvalue(r, 0, typ, bspline_ind, param_ind)
dw = find_shapegrad_dirichlet(r, omega, p, p_adj, geom, ds, c, typ, bspline_ind, param_ind, tie, ep_list)
x_points = []
y_points = []
omegas = [omega.real]


if cache:
    #1e-2
    x_points = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05, 3.6e-05, 4.9000000000000005e-05, 6.4e-05, 8.100000000000002e-05]
    y_points = [1.5595741017592043e-07, 4.032631932177923e-07, 7.419609382579057e-07, 1.1720353184454016e-06, 1.693486987923605e-06, 2.306118346090077e-06, 3.010134360987302e-06, 3.805322903190143e-06, 4.6918116816531234e-06]
    delta_ws = [-0.10918531041603785, -0.1090939620431719, -0.10900257008117364, -0.10891119344602629, -0.10881981615673553, -0.10872863646804731, -0.10863725181131656, -0.10854607928401094, -0.1084547790477508]


    #1e-2 ep 1e-8
    # x = [1.0000000000000001e-16, 4.0000000000000004e-16, 9.000000000000002e-16, 1.6000000000000002e-15, 2.4999999999999996e-15, 3.600000000000001e-15, 4.900000000000001e-15, 6.400000000000001e-15, 8.1e-15]
    # y = [7.769495305920951e-13, 2.61571636193569e-12, 2.97973221317926e-11, 4.172600383913546e-12, 5.561029840549736e-12, 1.3994676017761073e-11, 3.7941091893861105e-11, 6.803767120437525e-12, 1.1877692929618557e-11]
    # delta_ws = [-1.0941896277927299e-06, -1.0952514450934814e-06, -1.120594284031995e-06, -1.0594427557464314e-06, -1.092024248805501e-06, -1.1129683841204496e-06, -1.1173590941382372e-06, -1.0486678192478394e-06, -1.0883387524529553e-06]
else:
    for i in range(1, 10):
        epsilon = ep_step*i
        omega_new = find_eigenvalue(r, epsilon*ep_list, typ, bspline_ind, param_ind)[0]
        Delta_w_FD = omega_new.real - omega.real
        x_points.append(epsilon**2)
        y_points.append(abs(Delta_w_FD - dw*epsilon))
        omegas.append(omega_new.real)

        print(f"x = {x_points}")
        print(f"y = {y_points}")
        print(f"delta_ws = {[(omegas[i] - omegas[i-1])/ep_step for i in range(1, len(omegas))]}")


plt.plot([x_points[0], x_points[-1]], [y_points[0], y_points[-1]], color='0.8', linestyle='--')
plt.plot(x_points, y_points)
plt.xlabel('$\epsilon^2$')
plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
plt.show()

print(f"\n\n\n\n\n THIS IS DW: {dw}\n\n\n\n")