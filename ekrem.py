from bspline.nurbs import *
from bspline.nurbs_geometry import *
import dolfinx.io

def cylinder(ro, ri, l, lc):
    points = [
        [[ro, 0, 0],
        [ro, ro, 0],
        [0, ro, 0],
        [-ro, ro, 0],
        [-ro, 0, 0],
        [-ro, -ro, 0],
        [0, -ro, 0],
        [ro, -ro, 0],
        [ro, 0, 0]],
        [[ro, 0, l],
        [ro, ro, l],
        [0, ro, l],
        [-ro, ro, l],
        [-ro, 0, l],
        [-ro, -ro, l],
        [0, -ro, l],
        [ro, -ro, l],
        [ro, 0, l]]
    ]

    weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]]
    knotsU = [0, 1/4, 1/2, 3/4, 1]
    knotsV = [0, 1]
    multiplicitiesU = [3, 2, 2, 2, 3]
    multiplicitiesV = [2, 2]

    cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    cylinder.set_uniform_lc(lc)

    points = [
        [[ri, 0, 0],
        [ri, ri, 0],
        [0, ri, 0],
        [-ri, ri, 0],
        [-ri, 0, 0],
        [-ri, -ri, 0],
        [0, -ri, 0],
        [ri, -ri, 0],
        [ri, 0, 0]],
        [[ri, 0, l],
        [ri, ri, l],
        [0, ri, l],
        [-ri, ri, l],
        [-ri, 0, l],
        [-ri, -ri, l],
        [0, -ri, l],
        [ri, -ri, l],
        [ri, 0, l]]
    ]
    inner_cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    inner_cylinder.set_uniform_lc(lc)

    points = [
        [[ro, 0, 0],
        [ro, ro, 0],
        [0, ro, 0],
        [-ro, ro, 0],
        [-ro, 0, 0],
        [-ro, -ro, 0],
        [0, -ro, 0],
        [ro, -ro, 0],
        [ro, 0, 0]],
        [[ri, 0, 0],
        [ri, ri, 0],
        [0, ri, 0],
        [-ri, ri, 0],
        [-ri, 0, 0],
        [-ri, -ri, 0],
        [0, -ri, 0],
        [ri, -ri, 0],
        [ri, 0, 0]]
    ]

    start = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    start.set_uniform_lc(lc)

    points = [[
        [ro, 0, l],
        [ro, ro, l],
        [0, ro, l],
        [-ro, ro, l],
        [-ro, 0, l],
        [-ro, -ro, l],
        [0, -ro, l],
        [ro, -ro, l],
        [ro, 0, l]],
        [[ri, 0, l],
        [ri, ri, l],
        [0, ri, l],
        [-ri, ri, l],
        [-ri, 0, l],
        [-ri, -ri, l],
        [0, -ri, l],
        [ri, -ri, l],
        [ri, 0, l]]
    ]

    end = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    end.set_uniform_lc(lc)

    return [start, cylinder, end, inner_cylinder]

[start, cylinder, end, inner_cylinder] = cylinder(0.25, 0.2, 0.5, 0.02)
geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
geom.generate_mesh()
geom.add_nurbs_groups([[1, 2, 3, 4]])
geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=False)

geom.create_function_space("CG", 2)

geom.create_node_to_param_map()
C = geom.get_displacement_field("control point", [0, 0], [0, 1])
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disp_field.xdmf", "w") as xdmf:
    xdmf.write_mesh(geom.msh)
    xdmf.write_function(C)