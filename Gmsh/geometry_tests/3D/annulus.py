from bspline.nurbs_geometry import NURBs3DGeometry
from bspline.nurbs import NURBsSurface
from pathlib import Path
import os

points = [
    [[1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0]],
    [[1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [-1, 1, 1],
    [-1, 0, 1],
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [1, 0, 1]]
]

weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]]
knotsU = [0, 1/4, 1/2, 3/4, 1]
knotsV = [0, 1]
multiplicitiesU = [2, 2, 2, 2, 2]
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
cylinder.set_uniform_lc(1e-2)

points = [
    [[0.5, 0, 0],
    [0.5, 0.5, 0],
    [0, 0.5, 0],
    [-0.5, 0.5, 0],
    [-0.5, 0, 0],
    [-0.5, -0.5, 0],
    [0, -0.5, 0],
    [0.5, -0.5, 0],
    [0.5, 0, 0]],
    [[0.5, 0, 1],
    [0.5, 0.5, 1],
    [0, 0.5, 1],
    [-0.5, 0.5, 1],
    [-0.5, 0, 1],
    [-0.5, -0.5, 1],
    [0, -0.5, 1],
    [0.5, -0.5, 1],
    [0.5, 0, 1]]
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
inner_cylinder.set_uniform_lc(1e-2)

points = [
    [[1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0]],
    [[0.5, 0, 0],
    [0.5, 0.5, 0],
    [0, 0.5, 0],
    [-0.5, 0.5, 0],
    [-0.5, 0, 0],
    [-0.5, -0.5, 0],
    [0, -0.5, 0],
    [0.5, -0.5, 0],
    [0.5, 0, 0]]
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
start.set_uniform_lc(1e-2)

points = [[
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [-1, 1, 1],
    [-1, 0, 1],
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [1, 0, 1]],
    [[0.5, 0, 1],
    [0.5, 0.5, 1],
    [0, 0.5, 1],
    [-0.5, 0.5, 1],
    [-0.5, 0, 1],
    [-0.5, -0.5, 1],
    [0, -0.5, 1],
    [0.5, -0.5, 1],
    [0.5, 0, 1]]
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
end.set_uniform_lc(1e-2)

geom = NURBs3DGeometry([start, cylinder, end])
geom.generate_mesh(show_mesh=True)