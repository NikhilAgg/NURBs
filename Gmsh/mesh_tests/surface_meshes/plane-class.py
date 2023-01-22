from bspline.nurbs import NURBsSurface
from bspline.gmsh_utils import create_bspline_surface_mesh_file
from pathlib import Path
import os

points = [
    [0, 0, 0],
    [3, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
]

weights = [1, 1, 1, 1]
knotsU = [0, 1]
knotsV = [0,1]
multiplicitiesU = [2, 2]
multiplicitiesV = [2, 2]

plane = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 1,
    degreeV = 1
)

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/plane.msh"
file_path = folder_path + relative_path

create_bspline_surface_mesh_file(plane, file_path, show_mesh=True)