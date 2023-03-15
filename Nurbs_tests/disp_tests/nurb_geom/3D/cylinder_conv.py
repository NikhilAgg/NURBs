from bspline.nurbs_geometry import NURBs3DGeometry
from cylinder_splines import cylinder as cyl, split_cylinder
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem
from dolfinx_utils import utils
import ufl
import time

def func(ro, ri, l, lc, typ):
    if typ == "split":
        start, cylinder, cylinder2, end, inner_cylinder = split_cylinder(ro, ri, l, lc)
        geom = NURBs3DGeometry([[start, cylinder, cylinder2, end, inner_cylinder]])
    else:
        start, cylinder, end, inner_cylinder = cyl(ro, ri, l, lc)
        geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
    geom.generate_mesh(show_mesh=False)
    geom.add_nurbs_groups([[1, 2, 3, 4]])
    geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)
    geom.create_function_space("Lagrange", 2)
    geom.create_node_to_param_map()

    t = time.time()
    C = fem.Function(geom.V)
    if typ == "ro":
        for j in range(2):
            for i in range(8):
                C += np.dot(geom.get_displacement_field("control point", (0, 1), [j, i]), (np.r_[geom.bsplines[0][1].ctrl_points[j][i][0:2], [0]]/ro))
    elif typ == "l":
        for j in range(2):
            for i in range(8):
                C += geom.get_displacement_field("control point", (0, 2), [j, i])[2]
    elif typ == "ri":
        for j in range(2):
            for i in range(8):
                C += np.dot(geom.get_displacement_field("control point", (0, 3), [j, i]), (np.r_[geom.bsplines[0][1].ctrl_points[j][i][0:2], [0]]/ro))
    elif typ == "split":
        for j in range(2):
            for i in range(8):
                C += np.dot(geom.get_displacement_field("control point", (0, 1), [j, i]), (np.r_[geom.bsplines[0][1].ctrl_points[j][i][0:2], [0]]/ro))

        for j in range(1):
            for i in range(8):
                C += np.dot(geom.get_displacement_field("control point", (0, 2), [1, i]), (np.r_[geom.bsplines[0][2].ctrl_points[1][i][0:2], [0]]/ro))
            
    print(f"\n\nDisplacement Field Calculated - Time taken: {time.time()-t}")
    dA = fem.assemble_scalar(fem.form((C)*ufl.ds))

    geom.model.remove_physical_groups()
    geom.model.remove()
    del geom.gmsh
    return dA

ro = 2
ri = 1
l = 1
typ = "split"

cache = True

if typ == "ro":
    actual = np.abs(2*np.pi*ro*l)
    x = [-0.125, -0.25, -0.375, -0.5] 
    y = [0.8042230888714917, 0.5422693907791163, 0.3102932620530576, 0.05319004898065867]

elif typ == "l":
    actual = np.abs(np.pi*(ro**2 - ri**2))
    x = [-0.125, -0.25, -0.375, -0.5]
    y = [0.6905341056507055, 0.4016788890772987, 0.18176649250970353, -0.06170838247102487]

elif typ == "ri":
    actual = np.abs(2*np.pi*ri*l)
    x = [-0.125, -0.25, -0.375, -0.5, -0.625]
    y = [0.5042811516561916, 0.2512854891283184, -0.008486696719992744, -0.22868447930640237, -0.4431434503262087]

elif typ == "split":
    actual = np.abs(2*np.pi*ro*l)
    x = [-0.1666666666666666, -0.3333333333333333, -0.5, -0.6666666666666666] 
    y = [0.8051657133682183, 0.616811113569617, 0.3455492203869159, 0.011287673138985774]

if not cache:
    x= []
    y = []
    for i in range(1, 7):
        print(f"Iteration {i}")
        xi = 10**(-i/3)
        y.append(np.log10(np.abs(np.abs(func(ro, ri, l, xi, typ))-actual)))
        x.append(np.log10(xi**0.5))
        print(f"Finished: \n x = {x} \n y = {y}\n")

slopes = []

for i in range(1, len(x)):
    slopes.append((y[i-1] - y[i])/(x[i-1] - x[i]))
print(f"\n\nSlope: {slopes}\nAverage Slope: {np.mean(slopes)}")

plt.plot(x, y)
plt.xlabel("Log dx")
plt.ylabel("Log Error")
plt.legend()
plt.show()