from bspline.nurbs import NURBsCurve
from bspline.nurbs_geometry import NURBs2DGeometry
from pathlib import Path
import os

points = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [-1, 1, 0],
    [-1, 0, 0],
    [-1, -1, 0],
    [0, -1, 0],
    [1, -1, 0],
    [1, 0, 0]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [2, 2, 2, 2, 2]

circle = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle.set_uniform_lc(1.5e-2)

points = [
    [0.5, 0, 0],
    [0.5, 0.5, 0],
    [0, 0.5, 0],
    [-0.5, 0.5, 0],
    [-0.5, 0, 0],
    [-0.5, -0.5, 0],
    [0, -0.5, 0],
    [0.5, -0.5, 0],
    [0.5, 0, 0]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [2, 2, 2, 2, 2]

circle2 = NURBsCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)
circle2.set_uniform_lc(1e-2)

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/donut.msh"
file_path = folder_path + relative_path

geom = NURBs2DGeometry([[circle], [circle2]])
geom.generate_mesh(show_mesh=True)