from pathlib import Path
pth = Path(__file__).parent

import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
if str(pth) not in sys.path:
    sys.path.append(str(pth))

import ufl
from mpi4py import MPI
from dolfinx import fem
from petsc4py import PETSc
from helmholtz_x.passive_flame_x import *
from helmholtz_x.eigensolvers_x import eps_solver
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.petsc4py_utils import conjugate_function
from dolfinx_utils import *
from bspline.nurbs_geometry import NURBs3DGeometry
import dolfinx.io
from bspline.nurbs import NURBsCurve
import numpy as np

# Defining Circle NURB
r = 0.5
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
circle.set_uniform_lc(5e-2)