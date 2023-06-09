from pathlib import Path
pth = Path(__file__).parent

import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
if str(pth) not in sys.path:
    sys.path.append(str(pth))

import ufl
from mpi4py import MPI
from dolfinx import fem
from petsc4py import PETSc
from helmholtz_x.passive_flame_x import *
from helmholtz_x.eigensolvers_x import eps_solver
from helmholtz_x.eigenvectors_x import normalize_eigenvector, normalize_adjoint
from helmholtz_x.petsc4py_utils import conjugate_function
from dolfinx_utils import *
from bspline.nurbs_geometry import NURBs3DGeometry
from geoms import *
import dolfinx.io
import numpy as np


class WriteResults:
    def __init__(self, typ, bspline_ind, param_ind):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return 
        self.filename = pth / f"results/annulus/results_{typ}{bspline_ind}{param_ind}.py"

        try:
            self.read_lines()
        except FileNotFoundError:
            with open(self.filename, "w") as f:
                f.write("import matplotlib.pyplot as plt\n\n\n")
            self.lines = ""

    def read_lines(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return 
        with open(self.filename, "r") as f:
            self.lines = f.read()

    def new_test(self, ro, ri, l, ep_list, ep_step, lc):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return 
        self.read_lines()
        self.ro = ro
        self.ri = ri
        self.l = l
        self.ep_list = ep_list
        self.ep_step = ep_step
        self.lc = lc

    def write_results(self, x_points, y_points, omegas, dw, ep_step, lc):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return 
        
        delta_ws = [(omegas[i] - omegas[i-1])/ep_step for i in range(1, len(omegas))]
        with open(self.filename, "w") as f:
            f.write(self.lines)
            f.write(f"# ro = {self.ro}\t ri = {self.ri}\t l = {self.l}\t ep_list = {self.ep_list} \t ep_step = {self.ep_step}\t lc = {self.lc}\n")
            f.write(f"x_points = {x_points}\n")    
            f.write(f"y_points = {y_points}\n")    
            f.write(f"omegas = {omegas}\n")
            f.write(f"delta_ws = {delta_ws}\n")
            f.write(f"delta_ws = {[(delta_ws[i] - delta_ws[i-1]) for i in range(1, len(delta_ws))]}\n") 
            f.write(f"dw = {dw}\n")
            f.write(f"plt.plot(x_points, y_points, label = {lc})\n")
            f.write(f"plt.scatter(x_points, y_points)\n\n\n")
            


def create_mesh(ro, ri, l, epsilon, typ, bspline_ind, param_inds, lc, degree, ep_list):
    circ, _ = circle(ro, 0, lc)
    geom = NURBs2DGeometry([[circ]])
    for i, param_ind in enumerate(param_inds):
        geom.nurbs[bspline_ind[0]][bspline_ind[1]].lc[param_ind[0]] = lc
        geom = edit_param_2d(geom, typ, bspline_ind, param_ind, epsilon*ep_list[i])
    geom.model.remove_physical_groups()
    geom.model.remove()
    geom.generate_mesh()
    geom.add_nurbs_groups([[1]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", degree)

    return geom


def normalize_robin_vectors(omega, A, C, p, p_adj, c, geom):
    L_dash = (2*omega)*C
    Lp = fem.Function(geom.V)
    L_dash.mult(p.vector, Lp.vector)

    first_term_val = conjugate_function(p_adj).vector.dot(Lp.vector)

    Z=1
    second_term = fem.form((1j*c/Z) * ufl.inner(p_adj, p)*ufl.ds)
    second_term_val = fem.assemble_scalar(second_term)

    norm_const = first_term_val + second_term_val

    p = p/np.sqrt(norm_const)
    p_adj = p_adj/np.sqrt(norm_const)

    return [p, p_adj]


def find_eigenvalue(ro, ri, l, lc, epsilon, typ, bspline_ind, param_ind, ep_list):
    degree = 2
    c_const = np.sqrt(1)
    geom = create_mesh(ro, ri, l, epsilon, typ, bspline_ind, param_ind, lc, degree, ep_list)
    c = fem.Constant(geom.msh, PETSc.ScalarType(c_const))

    boundary_conditions = {1: {'Dirichlet'}}
    ds = ufl.ds
    facet_tags = geom.facet_tags

    matrices = PassiveFlame(geom.msh, facet_tags, boundary_conditions, c , degree = degree)
    matrices.assemble_A()
    matrices.assemble_B()
    matrices.assemble_C()
    A = matrices.A
    B = matrices.B
    C = matrices.C

    target =  c_const * np.pi * 1.2
    E = eps_solver(A, C, target**2, nev = 1, two_sided=True)
    omega, p = normalize_eigenvector(geom.msh, E, 0, degree=degree, which='right')
    _, p_adj = normalize_eigenvector(geom.msh, E, 0, degree=degree, which='left')

    # p = normalize_magnitude(p)
    # p_adj = normalize_magnitude(p_adj)
    
    p, p_adj = normalize_robin_vectors(omega, A, C, p, p_adj, c, geom)
    # p_adj = normalize_adjoint(omega, p, p_adj, matrices)
    # plot_slices(geom.msh, geom.V, p)

    return [omega, p, p_adj, geom, ds, c]


def find_shapegrad_dirichlet(p, p_adj, geom, ds, c, typ, bspline_ind, param_inds, ep_list):
    G = -c**2*ufl.Dn(p)*ufl.Dn(p_adj)

    geom.create_function_space("Lagrange", 1)
    geom.create_node_to_param_map()
    C = fem.Function(geom.V)
    dw = 0
    for i, param_ind in enumerate(param_inds):
        C += np.dot(geom.get_displacement_field(typ, bspline_ind, param_ind, tie=False), ep_list[i])
    
    # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disp_field.xdmf", "w") as xdmf:
    #     # C = geom.get_displacement_field(typ, bspline_ind, param_inds[0], tie=True, flip_norm=True)
    #     xdmf.write_mesh(geom.msh)
    #     xdmf.write_function(p)

    dw = fem.assemble_scalar(fem.form(G*C*ds))

    return dw


def taylor_test(ro, ri, l, lc, bspline_ind, param_ind, typ, ep_step, ep_list, fil):
    omega, p, p_adj, geom, ds, c = find_eigenvalue(ro, ri, l, lc, 0, typ, bspline_ind, param_ind, [0]*len(param_ind))
    dw = find_shapegrad_dirichlet(p, p_adj, geom, ds, c, typ, bspline_ind, param_ind, ep_list)
    # print(f"DW: {dw}")
    x_points = [0]
    y_points = [0]
    omegas = [omega.real]

    for i in range(1, 5):
        epsilon = ep_step*i 
        omega_new = find_eigenvalue(ro, ri, l, lc, epsilon, typ, bspline_ind, param_ind, ep_list)[0]
        
        Delta_w_FD = omega_new.real - omega.real
        x_points.append(epsilon**2)
        y_points.append(abs(Delta_w_FD - dw*epsilon))
        omegas.append(omega_new.real)

        fil.write_results(x_points, y_points, omegas, dw, ep_step, lc)
        

ep_step = 0.01
ro = 0.5
ri = 0.25
l = 1
lcs = [5e-3]

bspline_inds = [(0, 0)] #, (0, 2)]
param_inds = [[[2]]] #1, 4 for control point -1, 1, 1 and 0, 5 for weight [[1, 4]]] #, 
typs = ['weight']

for lc in lcs:
    for i in range(len(typs)):
        if typs[i] == "control point":
            ep_list = np.array([[0., 1.]])
        elif typs[i] == "weight":
            ep_list = [1]

        fil = WriteResults(typs[i], bspline_inds[i], param_inds[i])
        
        fil.new_test(ro, ri, l, ep_list, ep_step, lc)
        taylor_test(ro, ri, l, lc, bspline_inds[i], param_inds[i], typs[i], ep_step, ep_list, fil)