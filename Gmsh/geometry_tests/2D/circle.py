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
circle.set_uniform_lc(1e-2)

folder_path = str(Path(__file__).parent)
if not os.path.isdir(folder_path + '/meshes'):
    os.mkdir(folder_path + "/meshes")
relative_path = "/meshes/circle.msh"
file_path = folder_path + relative_path

geom = NURBs2DGeometry([[circle]])
geom.generate_mesh(show_mesh=False)
geom.add_bspline_groups([[1]])
geom.model_to_fenics()
geom.create_function_space('Lagrange', 2)
geom.create_node_to_param_map()