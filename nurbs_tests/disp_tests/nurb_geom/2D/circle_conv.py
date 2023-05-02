from bspline.nurbs_geometry import NURBs2DGeometry
from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

def func(dx, r):
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
    circle.set_uniform_lc(dx)

    geom = NURBs2DGeometry([[circle]])
    geom.generate_mesh(show_mesh=False)
    # t = time.time()
    geom.add_nurbs_groups([[1]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", 2)
    geom.create_node_to_param_map()

    C = fem.Function(geom.V)
    C += np.dot(geom.get_displacement_field("control point", (0, 0), [0]), (np.array(geom.nurbs[0][0].ctrl_points[0])/r))
    for i in range(1, 8):
        C += np.dot(geom.get_displacement_field("control point", (0, 0), [i]), (np.array(geom.nurbs[0][0].ctrl_points[i])/r))

    dA = fem.assemble_scalar(fem.form((C)*ufl.ds))
    geom.model.remove_physical_groups()
    geom.model.remove()
    del geom.gmsh
    return dA


x= []
y = []
r = 2
cache = True

if cache:
    x = [0.31622776601683794, 0.1, 0.03162277660168379, 0.01]
    y = [-1.118554890584832, -2.2923902571323693, -3.451314008511964, -4.60415232440682]
else:
    for i in range(1, 5):
        print(i)
        xi = 10**(-i/2)
        y.append(np.log10(np.abs(np.abs(func(xi, r))-np.abs(2*np.pi*r))))
        x.append(xi)

print(x)
print(y)
plt.plot(x, y)
plt.xscale("log", base=10)
plt.xlabel("dx")
plt.ylabel("Log Error")
plt.legend()
plt.show()

slopes = []
for i in range(1, len(x)):
    slopes.append((y[i-1] - y[i])/(np.log10(x[i-1]) - np.log10(x[i])))
print(f"\n\nSlope: {slopes}\nAverage Slope: {np.mean(slopes)}")