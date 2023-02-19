from bspline.nurbs_geometry import NURBs2DGeometry
from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

r = 3
points = [
    [r, 0],
    [r, r],
    [0, r],
    [-r, r],
    [-r, 0],
    [-r, -r],
    [0, -r],
    [r, -r],
    [r, 0]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [3, 2, 2, 2, 3]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1e-3)

geom = NURBs2DGeometry([[circle]])
geom.generate_mesh(show_mesh=False)
t = time.time()
geom.add_bspline_groups([[1]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
geom.create_function_space("Lagrange", 2)
geom.create_node_to_param_map()

C = fem.Function(geom.V)
C += np.dot(geom.get_displacement_field("control point", (0, 0), [0]), (np.array(geom.bsplines[0][0].ctrl_points[0])/r))
for i in range(1, 8):
    C += np.dot(geom.get_displacement_field("control point", (0, 0), [i]), (np.array(geom.bsplines[0][0].ctrl_points[i])/r))

dA = fem.assemble_scalar(fem.form((C)*ufl.ds))
print(dA/(2*np.pi*r))