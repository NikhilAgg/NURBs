from bspline.nurbs import NURBsCurve
import numpy as np
import matplotlib.pyplot as plt

r = 1
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
circle.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 10)
print([circle.get_curvature(u) for u in t])