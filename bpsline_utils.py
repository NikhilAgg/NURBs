from bspline.nurbs import NURBsCurve, NURBsSurface
import numpy as np

def get_unit_normals(normal):
    v1 = np.cross(normal, [1, 0, 0])
    if np.all(v1 == 0):
        v1 = np.cross(normal, [0, 1, 0])

    v2 = np.cross(normal, v1)

    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)

    return [v1, v2]


def translate_bspline_curve(curve, vector):
    for i, point in enumerate(curve.ctrl_points):
        curve.ctrl_points[i] = point + vector

    
    return curve


def create_circle_bspline_curve(r=1, O=[0, 0, 0], normal=[0, 0, 1]):
    v1, v2 = get_unit_normals(normal)
    
    points = [
    O + r*v1,
    O + r*v1 + r*v2,
    O + r*v2,
    O - r*v1 + r*v2,
    O - r*v1,
    O - r*v1 - r*v2,
    O - r*v2,
    O + r*v1 - r*v2,
    O + r*v1
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

    return circle

def create_prism_bspline_surface(curve, L, nL):
    unit_norm = curve.get_unit_normal()

    dL = L/nL
    points = []
    for i in range(nL+1):
        section = translate_bspline_curve(curve, unit_norm*dL)
        points.extend(section.ctrl_points[:])

    knotsV = [i for i in range(nL+1)]
    multiplicitiesV = [2, *[1]*(nL-1), 2]
    
    surface = NURBsSurface(
        points,
        np.tile(curve.weights, nL+1),
        curve.knots,
        knotsV,
        curve.multiplicities,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )

    return surface

def create_cylinder_bspline_surface(r=1, O=[0, 0, 0], L=1, normal=[0, 0, 1], nL=1):
    unit_norm = normal/np.linalg.norm(normal)

    circle = create_circle_bspline_curve(r, O, unit_norm)
    surface = create_prism_bspline_surface(circle, L, nL)

    return surface
