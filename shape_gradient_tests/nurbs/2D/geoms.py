from bspline.nurbs import NURBsSurface, NURBsCurve
from bspline.nurbs_geometry import NURBs2DGeometry
import gmsh
import sys
from dolfinx.io import gmshio
from mpi4py import MPI


def unit_square(epsilon, lc):
    points = [
        [0., 0 - epsilon],
        [1., 0. - epsilon],
        [1., 1.],
        [0., 1.],
        [0., 0. - epsilon]
    ]

    weights = [1, 1, 1, 1, 1]
    knots = [0, 1, 2, 3, 4]
    multiplicities = [2, 1, 1, 1, 2]

    square = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        1
    )
    square.set_uniform_lc(lc)

    return square

def unit_square2(l, epsilon, lc):
    points = [
        [-epsilon, -epsilon],
        [l+epsilon, -epsilon],
        [l+epsilon, l+epsilon],
        [-epsilon, l+epsilon],
        [-epsilon, -epsilon]
    ]

    weights = [1, 1, 1, 1, 1]
    knots = [0, 1, 2, 3, 4]
    multiplicities = [2, 1, 1, 1, 2]

    square = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        1
    )
    square.set_uniform_lc(lc)

    return square

# square = unit_square(0.1, 1e-2)
# geom = NURBs2DGeometry([[square]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[9]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)


def split_unit_square(epsilon, lc):
    points = [
        [0., 0 - epsilon],
        [1., 0. - epsilon]
    ]

    weights = [1, 1]
    knots = [0, 1]
    multiplicities = [2, 2]

    edge1 = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        1
    )
    edge1.set_uniform_lc(lc)

    points = [
        [1., 0. - epsilon],
        [1., 1.]
    ]

    edge2 = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        1
    )
    edge2.set_uniform_lc(lc)

    points = [
        [1., 1.],
        [0., 1.]
    ]

    edge3 = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        1
    )
    edge3.set_uniform_lc(lc)

    points = [
        [0., 1.],
        [0., 0. - epsilon]
    ]

    edge4 = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        1
    )
    edge4.set_uniform_lc(lc)

    return [edge1, edge2, edge3, edge4]

# edge1, edge2, edge3, edge4 = split_unit_square(1, 1e-2)
# geom = NURBs2DGeometry([[edge1, edge2, edge3, edge4]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[10, 11, 12, 13]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)

def circle(r, epsilon, lc):
    r = r + epsilon
    points = [
        [r, 0],
        [r, r],
        [0, r],
        [-r, r],
        [-r, 0],
        [-r, -r],
        [0, -r],
        [r, -r],
        [r, 0]
    ]

    weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
    knots = [0, 1/4, 1/2, 3/4, 1]
    multiplicities = [3, 2, 2, 2, 3]

    circle = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        2
    )
    circle.set_uniform_lc(lc)
    circle.flip_norm = True
    return circle

# circle = circle(1, 0, 1e-2)
# geom = NURBs2DGeometry([[circle]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[10]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)

class FauxGeom():
    def init(self):
        self.gmsh = None
        self.V = None
        self.msh = None
        self.facet_tags = None


def circle2(r, epsilon, lc, show_mesh=False):
    geom = FauxGeom()
    geom.gmsh = gmsh
    geom.gmsh.initialize()
    geom.gmsh.model.remove_physical_groups()
    geom.gmsh.model.remove()

    gmsh.option.set_number("Mesh.MeshSizeFactor", lc)
    circle = geom.gmsh.model.occ.add_circle(0, 0, 0, r+epsilon)
    circle_loop = geom.gmsh.model.occ.add_curve_loop([circle])
    plane = gmsh.model.occ.add_plane_surface([circle_loop])
    geom.gmsh.model.occ.synchronize()
    geom.gmsh.model.add_physical_group(2, [plane], 9)
    geom.gmsh.model.add_physical_group(1, [circle_loop], 9)
    geom.gmsh.model.mesh.generate(2)
    geom.msh, _, geom.facet_tags = gmshio.model_to_mesh(geom.gmsh.model, MPI.COMM_WORLD, 0)

    if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()

    return geom


# circle2(10, 0.5, 1e-1)

def edit_param_2d(geom, typ, bspline_ind, param_ind, epsilon):
    i, j = bspline_ind
    bspline = geom.nurbs[i][j]

    if typ == "control point":
        bsplines = geom.get_deriv_nurbs("control point", bspline_ind, param_ind)
        for bspline in bsplines:
            params = bsplines[bspline]
            for param_inds in params:
                    geom.nurbs[bspline[0]][bspline[1]].ctrl_points[param_inds[0]] += epsilon

    elif typ == "weight":
        bspline.weights[param_ind] += epsilon

    return geom

# circle = circle(1, 0, 1e-2)
# geom = NURBs2DGeometry([[circle]])
# edit_param_2d(geom, "weight", (0, 0), 0, 4)
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[10]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)

def thin(ro, ri, epsilon, lc):
    ro = ro + epsilon
    points = [
        [2*ro, 0],
        [2*ro, ro],
        [0, ro],
        [-2*ro, ro],
        [-2*ro, 0],
        [-2*ro, -ro],
        [0, -ro],
        [2*ro, -ro],
        [2*ro, 0]
    ]

    weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
    knots = [0, 1/4, 1/2, 3/4, 1]
    multiplicities = [3, 2, 2, 2, 3]

    circle = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        2
    )
    circle.set_uniform_lc(lc)
    circle.flip_norm = True

    dx = (ro - 0.75*ri)*0.75
    dy = (ro - ri) * 0.75
    points = [
        [1.25*ri + dx, 0 + dy],
        [1.25*ri + dx, ri + dy],
        [0 + dx, ri + dy],
        [-1.25*ri + dx, ri + dy],
        [-1.25*ri + dx, 0 + dy],
        [-1.25*ri + dx, -ri + dy],
        [0 + dx, -ri + dy],
        [1.25*ri+ dx, -ri + dy],
        [1.25*ri + dx, 0 + dy]
    ]
    inner_circle = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        2
    )
    inner_circle.set_uniform_lc(lc)
    inner_circle.flip_norm = False

    return circle, inner_circle

def smooth_thin(ro, ri, epsilon, lc):
    ro = ro + epsilon
    points = [
        [2*ro, 0],
        [2*ro, ro],
        [0, ro],
        [-2*ro, ro],
        [-2*ro, 0],
        [-2*ro, -ro],
        [0, -ro],
        [2*ro, -ro],
        [2*ro, 0]
    ]

    weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
    knots = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    multiplicities = [4, 1, 1, 1, 1, 1, 4]

    circle = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        3
    )
    circle.set_uniform_lc(lc)
    circle.flip_norm = True

    dx = (ro - 0.75*ri)*0.75
    dy = (ro - ri) * 0.75
    points = [
        [1.25*ri + dx, 0 + dy],
        [1.25*ri + dx, ri + dy],
        [0 + dx, ri + dy],
        [-1.25*ri + dx, ri + dy],
        [-1.25*ri + dx, 0 + dy],
        [-1.25*ri + dx, -ri + dy],
        [0 + dx, -ri + dy],
        [1.25*ri+ dx, -ri + dy],
        [1.25*ri + dx, 0 + dy]
    ]
    inner_circle = NURBsCurve(
        points,
        weights,
        knots,
        multiplicities,
        3
    )
    inner_circle.set_uniform_lc(lc)
    inner_circle.flip_norm = False

    return circle, inner_circle
