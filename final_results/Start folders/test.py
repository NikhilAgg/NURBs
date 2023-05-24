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

def circle(ro, ri, l, lc, x=1, y=1):
    rx = ro*x
    ry = ro*y
    points = [
        [[rx, 0, l], 
         [rx, ry, l],
         [0, ry, l],
         [-rx, ry, l],
         [-rx, 0, l]],
        [[rx/2, 0, l], 
         [rx/2, ry/2, l],
         [0, ry/2, l],
         [-rx/2, ry/2, l],
         [-rx/2, 0, l]],
        [[rx/3, 0, l],
         [rx/6, 0, l],
         [0, 0, l],
         [-rx/6, 0, l],
         [-rx/3, 0, l]]
    ]

    weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1, 1, 1, 1]]
    knotsU = [0, 1/4, 1/2]
    knotsV = [0, 1, 2]
    multiplicitiesU = [3, 2, 3]
    multiplicitiesV = [2, 1, 2]

    top = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    top.set_uniform_lc(lc)

    return top

def annulus(ro, ri, l, lc, x=1, y=1):
    rx = ro*x
    ry = ro*y
    points = [
        [[rx, 0, l], 
         [rx, ry, l],
         [0, ry, l],
         [-rx, ry, l],
         [-rx, 0, l]],
        [[rx/2, 0, l], 
         [rx/2, ry/2, l],
         [0, ry/2, l],
         [-rx/2, ry/2, l],
         [-rx/2, 0, l]]
    ]

    weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1]]
    knotsU = [0, 1/4, 1/2]
    knotsV = [0, 1]
    multiplicitiesU = [3, 2, 3]
    multiplicitiesV = [2, 2]

    top = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    top.set_uniform_lc(lc)

    return top

def bottom(ro, l, lc):
    points = [[[ro, 0, 0],
              [-ro, 0, 0]],
              [[ro, 0, l],
              [-ro, 0, l]]]
    
    weights = [[1, 1], [1, 1]]
    knotsU = [0, 1]
    knotsV = [0, 1]
    multiplicitiesU = [2, 2]
    multiplicitiesV = [2, 2]

    bot = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 1,
        degreeV = 1
    )
    bot.set_uniform_lc(lc)

    points = [
        [[ro, 0, 0],
        [ro, ro, 0],
        [0, ro, 0],
        [-ro, ro, 0],
        [-ro, 0, 0]],
        [[ro, 0, l],
        [ro, ro, l],
        [0, ro, l],
        [-ro, ro, l],
        [-ro, 0, l]]
    ]

    weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1]]
    knotsU = [0, 1/4, 1/2]
    knotsV = [0, 1]
    multiplicitiesU = [3, 2, 3]
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

    return [cylinder, bot]

[start, cylinder, end, inner_cylinder] = cylinder(5, 2.5, 1, 0.5)
top1 = circle(5, 0, 0, 0.5)
top2 = circle(5, 0, 0, 0.5, 1, -1)
top3 = circle(5, 0, 1, 0.5)
top4 = circle(5, 0, 1, 0.5, 1, -1)

# top1 = annulus(5, 0, 0, 0.5)
# top2 = annulus(5, 0, 0, 0.5, 1, -1)
# top3 = annulus(5, 0, 1, 0.5)
# top4 = annulus(5, 0, 1, 0.5, 1, -1)

[cylinder, bot] = bottom(5, 1, 0.5)

geom = NURBs3DGeometry([[top1, cylinder, top3, bot]])
geom.generate_mesh()
geom.add_nurbs_groups([[1, 2, 3, 4]])
gmsh.fltk.run()

# geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
# geom.generate_mesh()
# geom.add_nurbs_groups([[1, 2, 3, 4]])

geom.model_to_fenics(MPI.COMM_WORLD, 0, show_mesh=True)

# geom.create_function_space("CG", 2)

# geom.create_node_to_param_map()
# C = geom.get_displacement_field("control point", [0, 0], [0, 1])
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disp_field.xdmf", "w") as xdmf:
#     xdmf.write_mesh(geom.msh)
#     xdmf.write_function(C)