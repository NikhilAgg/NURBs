from bspline.nurbs import NURBsCurve
from bspline.gmsh_utils import create_bspline_curve_mesh_file
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

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/circle.msh"
file_path = folder_path + relative_path

create_bspline_curve_mesh_file(circle, file_path, show_mesh=True)