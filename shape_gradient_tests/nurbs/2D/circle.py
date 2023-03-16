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


def create_mesh(r, epsilon, lc, degree):
    square = circle(r, epsilon, lc)
    geom = NURBs2DGeometry([[square]])
    geom.model.remove_physical_groups()
    geom.model.remove()
    geom.generate_mesh()
    geom.add_nurbs_groups([[9]])
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


def find_eigenvalue(r, epsilon):
    degree = 3
    c_const = np.sqrt(1)
    geom = create_mesh(r, epsilon, 1e-1, degree)
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


def find_shapegrad_dirichlet(r, omega, p, p_adj, geom, ds, c):
    G = -c**2*ufl.Dn(p)*ufl.Dn(p_adj)
    C = fem.Function(geom.V)
    geom.create_node_to_param_map()
    for i in range(8):
        C += np.dot(geom.get_displacement_field("control point", (0, 0), [i], flip=True), (np.array(geom.nurbs[0][0].ctrl_points[i])/r))

    dw = fem.assemble_scalar(fem.form(G*C*ds))

    return dw


cache = True
ep_step = 0.001
r = 1


omega, p, p_adj, geom, ds, c = find_eigenvalue(r, 0)
dw = find_shapegrad_dirichlet(r, omega, p, p_adj, geom, ds, c)
x_points = []
y_points = []
omegas = [omega.real]
print(f"\n\n\n\n\n THIS IS DW: {dw}\n\n\n\n")


if cache:
    # #1e-1
    # x = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05, 3.6e-05, 4.9000000000000005e-05, 6.4e-05, 8.100000000000002e-05]
    # y = [2.594510417236809e-05, 5.671992910494944e-05, 2.9430358512060745e-05, 6.982335654349969e-05, 0.00011497357904706809, 0.00016486493911673644, 0.00021947309354771946, 0.0002788165260208253, 0.00034284961498329963]
    # delta_ws = [-2.4044751543783605, -2.399645433618147, -2.4577098291436172, -2.3900272605192896, -2.38527003604716, -2.38052889848106, -2.3758121041197455, -2.3710768260776227, -2.3663871695882577]
    

    # #1e-1 with C
    # x_points = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05, 3.6e-05, 4.9000000000000005e-05, 6.4e-05, 8.100000000000002e-05]
    # y_points = [1.2622069023083583e-05, 2.0414417285953904e-05, 8.627116107429427e-05, 8.4445336238307e-05, 7.78622869301894e-05, 6.653810005597359e-05, 5.049711882044311e-05, 2.972085954278808e-05, 4.254943775768016e-06]

    # #2e-2
    # x = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05, 3.6e-05, 4.9000000000000005e-05, 6.4e-05, 8.100000000000002e-05]
    # y = [9.640006767040483e-06, 2.4076166141592283e-05, 4.159301997168092e-05, 6.558011396408554e-05, 9.431732526576052e-05, 0.0001277956934545247, 0.00016600027844299037, 0.00020724279953235683, 0.00025485362129385944]

    # #1e-2
    x_points = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05, 3.6e-05, 4.9000000000000005e-05, 6.4e-05, 8.100000000000002e-05]
    y_points = [4.826942040988928e-06, 1.438578614194018e-05, 2.8725946768262157e-05, 4.7895966754595704e-05, 7.175578845368229e-05, 0.00010035429232609257, 0.00013373921033287342, 0.00017177241692258732, 0.00021456366712740635]

    #1e-2 with C
    # x_points = [1e-06, 4e-06, 9e-06, 1.6e-05, 2.5e-05, 3.6e-05, 4.9000000000000005e-05, 6.4e-05, 8.100000000000002e-05]
    # y_points = [9.997600288066087e-07, 6.73142211757554e-06, 1.7244400731715633e-05, 3.2587238705866425e-05, 5.261987839277199e-05, 7.739120025299952e-05, 0.00010694893624759935, 0.00014115496082512877, 0.00018011902901776677]
    # delta_ws = [-2.402443175266278, -2.3977112732063155, -2.3929299566809448, -2.388100097320933, -2.3834102956081793, -2.3786716134348573, -2.373885199300485, -2.3692369107175537, -2.3644788671024486]

else:
    for i in range(1, 10):
        epsilon = 0.001*i
        omega_new = find_eigenvalue(r, epsilon)[0]
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