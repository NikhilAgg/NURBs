from bspline.bspline_curve import BSplineCurve
from bspline.gmsh_utils import create_bspline_curve_mesh_file
from pathlib import Path
import os

points = [
    [1, 0, 0],
    [3, 1, 0],
    [1, 2, 0],
    [0, 1, 0],
    [-1, 2, 0],
    [-2, 1, 0],
    [-1, 0, 0],
    [-3, -1, 0],
    [-1, -2, 0],
    [0, -1, 0],
    [1, -2, 0],
    [2, -1, 0],
    [1, 0, 0]
]

weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
knots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
multiplicities = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]

circle = BSplineCurve(
    points,
    weights,
    knots,
    multiplicities,
    2
)

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/curve.msh"
file_path = folder_path + relative_path

create_bspline_curve_mesh_file(circle, file_path, show_mesh=True)