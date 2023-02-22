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

r=5
j = 1
lc = 1e-1

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
geom.add_bspline_groups([[1]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
geom.create_function_space("Lagrange", 3)
geom.create_node_to_param_map()
ds = ufl.Measure("ds", domain=geom.msh, subdomain_data=geom.facet_markers)

Ones = fem.Function(geom.V)
Ones.interpolate(lambda x: x[0]*0+1)
area = fem.assemble_scalar(fem.form((Ones)*ufl.dx))

Cp = geom.get_displacement_field("control point", (0, 0), [j])
Cw = geom.get_displacement_field("weight", (0, 0), [j])

geom.model.remove_physical_groups()
geom.model.remove()

print("\n\nTest Initialised\n-----------------------------------")


def func(epsilon_p, epsilon_w, j):
    points2 = copy.deepcopy(points)
    points2[j] = [sum(x) for x in zip(points[j], epsilon_p)]
    circle = NURBsCurve(
        points2,
        weights,
        knots,
        multiplicities,
        2
    )
    circle.set_uniform_lc(lc)
    geom = NURBs2DGeometry([[circle]])
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

    weights2 = copy.deepcopy(weights)
    weights2[j] += epsilon_w
    circle = NURBsCurve(
        points,
        weights2,
        knots,
        multiplicities,
        2
    )
    circle.set_uniform_lc(lc)
    geom = NURBs2DGeometry([[circle]])
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
ep_step = 0.00001
cache = False

if cache:
    pass
else:
    for i in range(1, 5):
        print(f"\n\nInteration {i}\n-----------------------------------")
        epsilon_w = ep_step*i
        epsilon_p = [0, epsilon_w]
        dA_p, dA_w = func(epsilon_p, epsilon_w, j)

        dAp_from_dp = fem.assemble_scalar(fem.form((Cp[0]*epsilon_p[0])*ds)) + fem.assemble_scalar(fem.form((Cp[1]*epsilon_p[1])*ds))
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