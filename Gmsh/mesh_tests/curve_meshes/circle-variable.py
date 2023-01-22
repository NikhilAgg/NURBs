from bspline.nurbs import NURBsCurve
from bspline.gmsh_utils import create_bspline_curve_mesh_model
from bspline.bspline_utils import create_circle_bspline_curve
from pathlib import Path
import os

circle = create_circle_bspline_curve(3, [1, 2, 3], [1, 1, 1])

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/circle.msh"
file_path = folder_path + relative_path

create_bspline_curve_mesh_model(circle, lc=2e-2, show_mesh=True)