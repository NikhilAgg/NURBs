from bspline.nurbs_geometry import NURBs2DGeometry
from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time
import copy

r=1
j = 1
lc = 0.5

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
circle.set_uniform_lc(lc)

geom = NURBs2DGeometry([[circle]])
geom.generate_mesh(show_mesh=False)
geom.add_nurbs_groups([[1]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
geom.create_function_space("Lagrange", 1)
geom.create_node_to_param_map()
ds = ufl.Measure("ds", domain=geom.msh, subdomain_data=geom.facet_markers)

def func(_, x):
    theta = np.arctan2(x[1], x[0])
    theta = theta if theta > 0 else theta + 2*np.pi
    return theta**2

def func2(x):
    f = []
    for coord in x.T:
        theta = np.arctan2(coord[1], coord[0])
        theta = theta if theta > 0 else theta + 2*np.pi
        f.append(theta**2)
    
    return f


theta2 = geom.function_from_params(1, func)
int_theta2 = fem.assemble_scalar(fem.form(theta2*ds))
theta22 = fem.Function(geom.V)
theta22.interpolate(func2)
int2_theta2 = fem.assemble_scalar(fem.form(theta22*ds))

geom.model.remove_physical_groups()
geom.model.remove()

print("\n\nTest Initialised\n-----------------------------------")

x= [0]
y1 = [0]
y2 = [0]
ep_step = 0.001
cache = False

if cache:
    pass
else:
    for i in range(1, 5):
        print(f"\n\nInteration {i}\n-----------------------------------")
        epsilon = ep_step*i
        dL = int_theta2*epsilon
        error = np.abs(np.abs(dL) - np.abs(8/3*np.pi**3*epsilon))

        print(f"Actual vs Calculated dAp: {dL} vs {int2_theta2*epsilon} - Error {error}")

        y1.append(error)
        x.append(epsilon**2)

print(x)
print(y1)
print(y2)
plt.plot(x, y1)
plt.xlabel("$\epsilon^2$")
plt.ylabel("Error in calculated area")
plt.show()
