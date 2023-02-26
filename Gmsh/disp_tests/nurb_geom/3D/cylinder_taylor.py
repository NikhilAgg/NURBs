from bspline.nurbs_geometry import NURBs3DGeometry
from cylinder_splines import cylinder as cyl, split_cylinder
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

def change_param(geom, typ, bspline_ind, param_ind, epsilon):
    bsplines = geom.get_deriv_bsplines("control point", bspline_ind, param_ind)
    for bspline in bsplines:
        params = bsplines[bspline]
        for param_inds in params:
            if typ == "control point":
                geom.bsplines[bspline[0]][bspline[1]].ctrl_points[param_inds[0]][param_inds[1]] += epsilon
            elif typ == "weight":
                geom.bsplines[bspline[0]][bspline[1]].weights[param_inds[0]][param_inds[1]] += epsilon

def get_weight_C(geom, bspline_ind, param_ind):
    Cw = fem.Function(geom.V)
    bsplines = geom.get_deriv_bsplines("control point", bspline_ind, param_ind)
    for bspline in bsplines:
        params = bsplines[bspline]
        for param_inds in params:
            Cw += geom.get_displacement_field("weight", bspline, param_inds)

    return Cw

ro=2.
ri = 1.
l = 1.

bspline_inds = (0, 1)
k = 1
j = 1
lc = 1e-1

start, cylinder, end, inner_cylinder = cyl(ro, ri, l, lc)
geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
geom.generate_mesh(show_mesh=False)
geom.add_bspline_groups([[1]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
geom.create_function_space("Lagrange", 3)
geom.create_node_to_param_map()
ds = ufl.Measure("ds", domain=geom.msh, subdomain_data=geom.facet_markers)

Ones = fem.Function(geom.V)
Ones.interpolate(lambda x: x[0]*0+1)
area = fem.assemble_scalar(fem.form((Ones)*ufl.dx))

Cp = geom.get_displacement_field("control point", bspline_inds, [k, j])
Cw = get_weight_C(geom, bspline_inds, [k, j])

geom.model.remove_physical_groups()
geom.model.remove()

print("\n\nTest Initialised\n-----------------------------------")


def func(epsilon_p, epsilon_w, j):
    start, cylinder, end, inner_cylinder = cyl(ro, ri, l, lc)
    geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])#
    change_param(geom, "control point", bspline_inds, (k, j), epsilon_p)
    geom.generate_mesh(show_mesh=False)
    geom.add_bspline_groups([[1]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", 3)
    geom.create_node_to_param_map()
    Ones = fem.Function(geom.V)
    Ones.interpolate(lambda x: x[0]*0+1)
    area_p = fem.assemble_scalar(fem.form((Ones)*ufl.dx))
    geom.model.remove_physical_groups()
    geom.model.remove()

    start, cylinder, end, inner_cylinder = cyl(ro, ri, l, lc)
    geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
    change_param(geom, "weight", bspline_inds, (k, j), epsilon_w)
    geom.generate_mesh(show_mesh=False)
    geom.add_bspline_groups([[1]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", 3)
    geom.create_node_to_param_map()
    Ones = fem.Function(geom.V)
    Ones.interpolate(lambda x: x[0]*0+1)
    area_w = fem.assemble_scalar(fem.form((Ones)*ufl.dx))
    geom.model.remove_physical_groups()
    geom.model.remove()

    return abs(area_p - area), abs(area_w - area)


x= [0]
y1 = [0]
y2 = [0]
ep_step = 0.01
cache = False

if cache:
    pass
else:
    for i in range(1, 5):
        print(f"\n\nInteration {i}\n-----------------------------------")
        epsilon_w = ep_step*i
        epsilon_p = [0, epsilon_w, 0]
        dA_p, dA_w = func(epsilon_p, epsilon_w, j)

        dAp_from_dp = fem.assemble_scalar(fem.form((Cp[0]*epsilon_p[0] + Cp[1]*epsilon_p[1] + Cp[2]*epsilon_p[2])*ds))
        dAw_from_dw = fem.assemble_scalar(fem.form((Cw*epsilon_w)*ds))

        print(f"Actual vs Calculated dAp: {dA_p} vs {dAp_from_dp} - Error {np.abs(np.abs(dA_p) - np.abs(dAp_from_dp))}")
        print(f"Actual vs Calculated dAp: {dA_w} vs {dAw_from_dw} - Error {np.abs(np.abs(dA_w) - np.abs(dAw_from_dw))}")

        y1.append(np.abs(np.abs(dA_p) - np.abs(dAp_from_dp)))
        y2.append(np.abs(np.abs(dA_w) - np.abs(dAw_from_dw)))
        x.append(epsilon_w**2)

print(x)
print(y1)
print(y2)
plt.plot(x, y1)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Error in calculated area")
plt.show()

plt.plot(x, y2)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Error in calculated area")
plt.show()