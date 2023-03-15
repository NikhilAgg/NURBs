from bspline.nurbs import NURBsSurface
from bspline.gmsh_utils import create_bspline_surface_mesh_model
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
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [-1, 1, 1],
    [-1, 0, 1],
    [-1, -1, 1],
    [0, -1, 1],
    [1, -1, 1],
    [1, 0, 1],
    [1, 0, 2],
    [1, 1, 2],
    [0, 1, 2],
    [-1, 1, 2],
    [-1, 0, 2],
    [-1, -1, 2],
    [0, -1, 2],
    [1, -1, 2],
    [1, 0, 2]
]

weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]*3
knotsU = [0, 1/4, 1/2, 3/4, 1]
knotsV = [0, 1, 2]
multiplicitiesU = [2, 2, 2, 2, 2]
multiplicitiesV = [2, 1, 2]

plane = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 2,
    degreeV = 1
)

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/cylinder.msh"
file_path = folder_path + relative_path

create_bspline_surface_mesh_model(plane, lc=2e-1, show_mesh=True)