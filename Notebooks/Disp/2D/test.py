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
from geoms import split_unit_square, unit_square
from bspline.nurbs_geometry import NURBs2DGeometry

def create_mesh1(epsilon, lc, degree):
    square = unit_square(epsilon, lc)
    geom = NURBs2DGeometry([[square]])
    geom.model.remove_physical_groups()
    geom.model.remove()
    geom.generate_mesh()
    geom.add_bspline_groups([[9]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", degree)

    return geom


def create_mesh(epsilon, lc, degree):
    edge1, edge2, edge3, edge4 = split_unit_square(epsilon, lc)
    geom = NURBs2DGeometry([[edge1, edge2, edge3, edge4]])
    geom.model.remove_physical_groups()
    geom.model.remove()
    geom.generate_mesh()
    geom.add_bspline_groups([[10, 11, 12, 13]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", degree)

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


def find_eigenvalue(epsilon):
    degree = 3
    c_const = np.sqrt(5)
    geom = create_mesh(epsilon, 1e-2, degree)
    c = fem.Constant(geom.msh, PETSc.ScalarType(c_const))

    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (3, lambda x: np.isclose(x[1], -epsilon)),
              (4, lambda x: np.isclose(x[1], 1))]
    
    ds, facet_tags = tag_boundaries(geom.msh, boundaries=boundaries, return_tags=True)

    boundary_conditions = {4: {'Dirichlet'},
                        3: {'Dirichlet'},
                        2: {'Dirichlet'},
                        1: {'Dirichlet'}}

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


def find_shapegrad_dirichlet(omega, p, msh, ds):
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    V2 = fem.VectorFunctionSpace(msh, ("Lagrange", 1))
    w = ufl.TrialFunction(V2)
    v = ufl.TestFunction(V2)

    a = ufl.inner(w, v)*ufl.dx
    L = ufl.inner(ufl.grad(p), v)*ufl.dx

    problem = fem.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    grad_p = problem.solve()

    G = fem.Function(V)
    expr = fem.Expression(2/omega.real*np.dot(grad_p, [0, -1])**2, V.element.interpolation_points())
    G.interpolate(expr)
    dw = fem.assemble_scalar(fem.form(-G*ds(3))).real

    return dw


def find_shapegrad_dirichlet(omega, p, p_adj, geom, ds, c):
    G = -c**2*ufl.Dn(p)*ufl.Dn(p_adj)
    C = fem.Function(geom.V)
    # geom = create_mesh1(0, 1e-2, 3)
    geom.create_node_to_param_map()
    C += geom.get_displacement_field("control point", (0, 0), [0])[1]
    C += geom.get_displacement_field("control point", (0, 0), [1])[1]
    print(f"\n\n\n\n CINT: {fem.assemble_scalar(fem.form(C*ds))}")
    C /= fem.assemble_scalar(fem.form(C*ds))
    dw = fem.assemble_scalar(fem.form(C* G*ds(3)))

    return dw

    
omega, p, p_adj, geom, ds, c = find_eigenvalue(0)
dw = find_shapegrad_dirichlet(omega, p, p_adj, geom, ds, c)
x_points = []
y_points = []
print(dw)

for i in range(1, 5):
    epsilon = 0.01*i
    omega_new = find_eigenvalue(epsilon)[0]
    Delta_w_FD = omega_new.real - omega.real
    x_points.append(epsilon**2)
    y_points.append(abs(Delta_w_FD - dw*epsilon))

plt.plot([x_points[0], x_points[-1]], [y_points[0], y_points[-1]], color='0.8', linestyle='--')
plt.plot(x_points, y_points)
plt.xlabel('$\epsilon^2$')
plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
