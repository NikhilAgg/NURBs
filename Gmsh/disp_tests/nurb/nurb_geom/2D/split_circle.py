from bspline.nurbs_geometry import NURBs2DGeometry
import numpy as np
import matplotlib.pyplot as plt
from split_circles import *
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

geom = NURBs2DGeometry([[circle1, circle2, circle3, circle4]])
geom.generate_mesh(show_mesh=False)
t = time.time()
geom.add_bspline_groups([[1, 2, 3, 4]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
geom.create_function_space("Lagrange", 2)
geom.create_node_to_param_map()

C = fem.Function(geom.V)
for i in range(4):
    C += np.dot(geom.get_displacement_field("control point", (0, i), [0]), (np.array(geom.bsplines[0][i].ctrl_points[0])/r))
    print(fem.assemble_scalar(fem.form((C)*ufl.ds)))
    C += np.dot(geom.get_displacement_field("control point", (0, i), [1]), (np.array(geom.bsplines[0][i].ctrl_points[1])/r))
    print(fem.assemble_scalar(fem.form((C)*ufl.ds)))
    # C += np.dot(geom.get_displacement_field("control point", (0, i), [2]), (np.array(geom.bsplines[0][i].ctrl_points[2])/r))

dA = fem.assemble_scalar(fem.form((C)*ufl.ds))
print(f"Final: {dA}")