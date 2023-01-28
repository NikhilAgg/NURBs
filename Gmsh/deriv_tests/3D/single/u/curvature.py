from bspline.nurbs import NURBsSurface
import numpy as np
import matplotlib.pyplot as plt

r = 5
points = [
    [[r, 0, 0],
    [r, r, 0],
    [0, r, 0],
    [-r, r, 0],
    [-r, 0, 0],
    [-r, -r, 0],
    [0, -r, 0],
    [r, -r, 0],
    [r, 0, 0]],
    [[r, 0, 1],
    [r, r, 1],
    [0, r, 1],
    [-r, r, 1],
    [-r, 0, 1],
    [-r, -r, 1],
    [0, -r, 1],
    [r, -r, 1],
    [r, 0, 1]]
]

weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]]
knotsU = [0, 1/4, 1/2, 3/4, 1]
knotsV = [0, 1]
multiplicitiesU = [3, 2, 2, 2, 3]
multiplicitiesV = [2, 2]

cylinder = NURBsSurface(
    points,
    weights,
    knotsU,
    knotsV,
    multiplicitiesU,
    multiplicitiesV,
    degreeU = 2,
    degreeV = 1
)
cylinder.set_uniform_lc(1e-2)

t = np.linspace(0, 1, 10)
s = np.linspace(0, 1, 10)
print([cylinder.get_curvature(u, v) for u, v in np.array(np.meshgrid(t, s)).T.reshape(-1,2)])