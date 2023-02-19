from bspline.nurbs_geometry import NURBs2DGeometry
import numpy as np
import matplotlib.pyplot as plt
from split_circles import *
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

def func(dx, r):
    circle1, circle2, circle3, circle4 = create_arcs(r, dx)
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
        C += np.dot(geom.get_displacement_field("control point", (0, i), [1]), (np.array(geom.bsplines[0][i].ctrl_points[1])/r))

    dA = fem.assemble_scalar(fem.form((C)*ufl.ds))

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
    y = [0.23764642496063035, -0.9316046745714642, -2.0731290115573557, -3.2212232579123534]
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