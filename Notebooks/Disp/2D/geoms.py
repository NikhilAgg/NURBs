from bspline.nurbs import NURBsSurface, NURBsCurve
from bspline.nurbs_geometry import NURBs2DGeometry


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

def circle(r, lc):
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

    return circle

# circle = circle(1, 1e-2)
# geom = NURBs2DGeometry([[circle]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[10]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)
