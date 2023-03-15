from bspline.nurbs import NURBsSurface
from bspline.gmsh_utils import create_bspline_surface_mesh_model
from bspline.bspline_utils import create_cylinder_bspline_surface
from pathlib import Path
import os

surface = create_cylinder_bspline_surface(2, [1, 2, 3], 5, [1, 1, 1], 5)

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/cylinder.msh"
file_path = folder_path + relative_path

create_bspline_surface_mesh_model(surface, lc=2e-1, show_mesh=True)