from bspline.nurbs_geometry import NURBs3DGeometry
from cylinder_splines import split_cylinder
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

ro = 2
ri = 1
l = 1
lc = 1e-1

start, cylinder, cylinder2, end, inner_cylinder = split_cylinder(ro, ri, l, lc)
geom = NURBs3DGeometry([[start, cylinder, cylinder2, end, inner_cylinder]])
geom.generate_mesh(show_mesh=False)
geom.add_bspline_groups([[1, 2, 3, 4, 5]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
geom.create_function_space("Lagrange", 2)
geom.create_node_to_param_map()

t = time.time()
C = fem.Function(geom.V)
old = 0
for j in range(2):
    for i in range(8):
        C += np.dot(geom.get_displacement_field("control point", (0, 1), [j, i]), (np.r_[geom.bsplines[0][1].ctrl_points[j][i][0:2], [0]]/ro))
        temp = fem.assemble_scalar(fem.form((C)*ufl.ds))
        print(temp-old)
        old = temp

for j in range(1):
    for i in range(8):
        C += np.dot(geom.get_displacement_field("control point", (0, 2), [1, i]), (np.r_[geom.bsplines[0][2].ctrl_points[1][i][0:2], [0]]/ro))
        temp = fem.assemble_scalar(fem.form((C)*ufl.ds))
        print(temp-old)
        old = temp
dA = fem.assemble_scalar(fem.form((C)*ufl.ds))
print(f"dA due to change in ro = {dA} - Time Taken: {time.time() - t}")

# print("Calculating dA corresponding to a change in length")
# t = time.time()
# C = fem.Function(geom.V)
# old = 0
# for j in range(2):
#     for i in range(8):
#         C += geom.get_displacement_field("control point", (0, 2), [j, i])[2]
#         temp = fem.assemble_scalar(fem.form((C)*ufl.ds))
#         print(temp-old)
#         old = temp
# dA = fem.assemble_scalar(fem.form((C)*ufl.ds))
# print(f"dA due to change in Length = {dA} - Time Taken: {time.time() - t}")
