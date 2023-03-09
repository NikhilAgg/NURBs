from pathlib import Path
pth = Path(__file__).parent
import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
if str(pth) not in sys.path:
    sys.path.append(str(pth))

from nurbs import NURBsSurface
import gmsh
import sys
from dolfinx.io import gmshio
from mpi4py import MPI

def annulus(ro0, ri0, l0, ro_ep, ri_ep, l_ep, lc):
    ro = ro0 + ro_ep
    ri = ri0 + ri_ep
    l = l0 + l_ep
    points = [
        [[2*ro, 0, 0],
        [2*ro, ro, 0],
        [0, ro, 0],
        [-2*ro, ro, 0],
        [-2*ro, 0, 0],
        [-2*ro, -ro, 0],
        [0, -ro, 0],
        [2*ro, -ro, 0],
        [2*ro, 0, 0]],
        [[ro, 0, l],
        [ro, 2*ro, l],
        [0, 2*ro, l],
        [-ro, 2*ro, l],
        [-ro, 0, l],
        [-ro, -2*ro, l],
        [0, -2*ro, l],
        [ro, -2*ro, l],
        [ro, 0, l]]
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
    cylinder.set_uniform_lc(lc)

    dx = (ro - 0.75*ri)*0.75
    dy = (ro - ri) * 0.75
    points = [
        [[1.25*ri + dx, 0 + dy, 0],
        [1.25*ri + dx, ri + dy, 0],
        [0 + dx, ri + dy, 0],
        [-1.25*ri + dx, ri + dy, 0],
        [-1.25*ri + dx, 0 + dy, 0],
        [-1.25*ri + dx, -ri + dy, 0],
        [0 + dx, -ri + dy, 0],
        [1.25*ri+ dx, -ri + dy, 0],
        [1.25*ri + dx, 0 + dy, 0]],
        [[ri + dy, 0 - dx, l],
        [ri + dy, 1.25*ri - dx, l],
        [0 + dy, 1.25*ri - dx, l],
        [-ri + dy, 1.25*ri - dx, l],
        [-ri + dy, 0 - dx, l],
        [-ri + dy, -1.25*ri - dx, l],
        [0 + dy, -1.25*ri - dx, l],
        [ri + dy, -1.25*ri - dx, l],
        [ri + dy, 0 - dx, l]]
    ]
    inner_cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    inner_cylinder.set_uniform_lc(lc)
    inner_cylinder.flip_norm = True

    points = [
        [[2*ro, 0, 0],
        [2*ro, ro, 0],
        [0, ro, 0],
        [-2*ro, ro, 0],
        [-2*ro, 0, 0],
        [-2*ro, -ro, 0],
        [0, -ro, 0],
        [2*ro, -ro, 0],
        [2*ro, 0, 0]],
        [[1.25*ri + dx, 0 + dy, 0],
        [1.25*ri + dx, ri + dy, 0],
        [0 + dx, ri + dy, 0],
        [-1.25*ri + dx, ri + dy, 0],
        [-1.25*ri + dx, 0 + dy, 0],
        [-1.25*ri + dx, -ri + dy, 0],
        [0 + dx, -ri + dy, 0],
        [1.25*ri + dx, -ri + dy, 0],
        [1.25*ri + dx, 0 + dy, 0]]
    ]

    start = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    start.set_uniform_lc(lc)
    start.flip_norm = True

    points = [[
        [ro, 0, l],
        [ro, 2*ro, l],
        [0, 2*ro, l],
        [-ro, 2*ro, l],
        [-ro, 0, l],
        [-ro, -2*ro, l],
        [0, -2*ro, l],
        [ro, -2*ro, l],
        [ro, 0, l]],
        [[ri + dy, 0 - dx, l],
        [ri + dy, 1.25*ri - dx, l],
        [0 + dy, 1.25*ri - dx, l],
        [-ri + dy, 1.25*ri - dx, l],
        [-ri + dy, 0 - dx, l],
        [-ri + dy, -1.25*ri - dx, l],
        [0 + dy, -1.25*ri - dx, l],
        [ri + dy, -1.25*ri - dx, l],
        [ri + dy, 0 - dx, l]]
    ]

    end = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    end.set_uniform_lc(lc)

    return [start, cylinder, end, inner_cylinder]

# start, cylinder, end, inner_cylinder = annulus(2, 1, 10, 0.5, -0.1, -0.4, 5e-1)
# geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[1, 2, 3, 4]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)


def edit_param_3d(geom, typ, bspline_ind, param_ind, epsilon):
    bsplines = geom.get_deriv_bsplines("control point", bspline_ind, param_ind)
    for bspline in bsplines:
        params = bsplines[bspline]
        for param_inds in params:
            if typ == "control point":
                geom.bsplines[bspline[0]][bspline[1]].ctrl_points[param_inds[0]][param_inds[1]] += epsilon
            elif typ == "weight":
                geom.bsplines[bspline[0]][bspline[1]].weights[param_inds[0]][param_inds[1]] += epsilon
            else:
                raise ValueError("Type is not defined")

    return geom

# start, cylinder, end, inner_cylinder = annulus(2, 1, 3, 0.5, -0.1, -0.4, 1e-1)
# geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
# geom = edit_param_3d(geom, "weight", (0, 1), [0, 1], 5)
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[1, 2, 3, 4]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)


def split_cylinder(ro, ri, l, lc):
    points = [
        [[ro, 0, 0],
        [ro, ro, 0],
        [0, ro, 0],
        [-ro, ro, 0],
        [-ro, 0, 0],
        [-ro, -ro, 0],
        [0, -ro, 0],
        [ro, -ro, 0],
        [ro, 0, 0]],
        [[ro, 0, l/2],
        [ro, ro, l/2],
        [0, ro, l/2],
        [-ro, ro, l/2],
        [-ro, 0, l/2],
        [-ro, -ro, l/2],
        [0, -ro, l/2],
        [ro, -ro, l/2],
        [ro, 0, l/2]]
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
    cylinder.set_uniform_lc(lc)

    points = [
        [[ro, 0, l/2],
        [ro, ro, l/2],
        [0, ro, l/2],
        [-ro, ro, l/2],
        [-ro, 0, l/2],
        [-ro, -ro, l/2],
        [0, -ro, l/2],
        [ro, -ro, l/2],
        [ro, 0, l/2]],
        [[ro, 0, l],
        [ro, ro, l],
        [0, ro, l],
        [-ro, ro, l],
        [-ro, 0, l],
        [-ro, -ro, l],
        [0, -ro, l],
        [ro, -ro, l],
        [ro, 0, l]]
    ]

    cylinder2 = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    cylinder2.set_uniform_lc(lc)

    points = [
        [[ri, 0, 0],
        [ri, ri, 0],
        [0, ri, 0],
        [-ri, ri, 0],
        [-ri, 0, 0],
        [-ri, -ri, 0],
        [0, -ri, 0],
        [ri, -ri, 0],
        [ri, 0, 0]],
        [[ri, 0, l],
        [ri, ri, l],
        [0, ri, l],
        [-ri, ri, l],
        [-ri, 0, l],
        [-ri, -ri, l],
        [0, -ri, l],
        [ri, -ri, l],
        [ri, 0, l]]
    ]
    inner_cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    inner_cylinder.set_uniform_lc(lc)

    points = [
        [[ro, 0, 0],
        [ro, ro, 0],
        [0, ro, 0],
        [-ro, ro, 0],
        [-ro, 0, 0],
        [-ro, -ro, 0],
        [0, -ro, 0],
        [ro, -ro, 0],
        [ro, 0, 0]],
        [[ri, 0, 0],
        [ri, ri, 0],
        [0, ri, 0],
        [-ri, ri, 0],
        [-ri, 0, 0],
        [-ri, -ri, 0],
        [0, -ri, 0],
        [ri, -ri, 0],
        [ri, 0, 0]]
    ]

    start = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    start.set_uniform_lc(lc)

    points = [[
        [ro, 0, l],
        [ro, ro, l],
        [0, ro, l],
        [-ro, ro, l],
        [-ro, 0, l],
        [-ro, -ro, l],
        [0, -ro, l],
        [ro, -ro, l],
        [ro, 0, l]],
        [[ri, 0, l],
        [ri, ri, l],
        [0, ri, l],
        [-ri, ri, l],
        [-ri, 0, l],
        [-ri, -ri, l],
        [0, -ri, l],
        [ri, -ri, l],
        [ri, 0, l]]
    ]

    end = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    end.set_uniform_lc(lc)

    return [start, cylinder, cylinder2, end, inner_cylinder]


def thin(ro0, ri0, l0, ro_ep, ri_ep, l_ep, lc):
    ro = ro0 + ro_ep
    ri = ri0 + ri_ep
    l = l0 + l_ep
    points = [
        [[2*ro, 0, 0],
        [2*ro, ro, 0],
        [0, ro, 0],
        [-2*ro, ro, 0],
        [-2*ro, 0, 0],
        [-2*ro, -ro, 0],
        [0, -ro, 0],
        [2*ro, -ro, 0],
        [2*ro, 0, 0]],
        [[2*ro, 0, l],
        [2*ro, ro, l],
        [0, ro, l],
        [-2*ro, ro, l],
        [-2*ro, 0, l],
        [-2*ro, -ro, l],
        [0, -ro, l],
        [2*ro, -ro, l],
        [2*ro, 0, l]]
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
    cylinder.set_uniform_lc(lc)

    dx = (ro - 0.75*ri)*0.75
    dy = (ro - ri) * 0.75
    points = [
        [[1.25*ri + dx, 0 + dy, 0],
        [1.25*ri + dx, ri + dy, 0],
        [0 + dx, ri + dy, 0],
        [-1.25*ri + dx, ri + dy, 0],
        [-1.25*ri + dx, 0 + dy, 0],
        [-1.25*ri + dx, -ri + dy, 0],
        [0 + dx, -ri + dy, 0],
        [1.25*ri+ dx, -ri + dy, 0],
        [1.25*ri + dx, 0 + dy, 0]],
        [[1.25*ri + dx, 0 + dy, l],
        [1.25*ri + dx, ri + dy, l],
        [0 + dx, ri + dy, l],
        [-1.25*ri + dx, ri + dy, l],
        [-1.25*ri + dx, 0 + dy, l],
        [-1.25*ri + dx, -ri + dy, l],
        [0 + dx, -ri + dy, l],
        [1.25*ri+ dx, -ri + dy, l],
        [1.25*ri + dx, 0 + dy, l]]
    ]
    inner_cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    inner_cylinder.set_uniform_lc(lc)
    inner_cylinder.flip_norm = True

    points = [
        [[2*ro, 0, 0],
        [2*ro, ro, 0],
        [0, ro, 0],
        [-2*ro, ro, 0],
        [-2*ro, 0, 0],
        [-2*ro, -ro, 0],
        [0, -ro, 0],
        [2*ro, -ro, 0],
        [2*ro, 0, 0]],
        [[1.25*ri + dx, 0 + dy, 0],
        [1.25*ri + dx, ri + dy, 0],
        [0 + dx, ri + dy, 0],
        [-1.25*ri + dx, ri + dy, 0],
        [-1.25*ri + dx, 0 + dy, 0],
        [-1.25*ri + dx, -ri + dy, 0],
        [0 + dx, -ri + dy, 0],
        [1.25*ri + dx, -ri + dy, 0],
        [1.25*ri + dx, 0 + dy, 0]]
    ]

    start = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    start.set_uniform_lc(lc)
    start.flip_norm = True

    points = [[
        [2*ro, 0, l],
        [2*ro, ro, l],
        [0, ro, l],
        [-2*ro, ro, l],
        [-2*ro, 0, l],
        [-2*ro, -ro, l],
        [0, -ro, l],
        [2*ro, -ro, l],
        [2*ro, 0, l]],
        [[1.25*ri + dx, 0 + dy, l],
        [1.25*ri + dx, ri + dy, l],
        [0 + dx, ri + dy, l],
        [-1.25*ri + dx, ri + dy, l],
        [-1.25*ri + dx, 0 + dy, l],
        [-1.25*ri + dx, -ri + dy, l],
        [0 + dx, -ri + dy, l],
        [1.25*ri + dx, -ri + dy, l],
        [1.25*ri + dx, 0 + dy, l]]
    ]

    end = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 2,
        degreeV = 1
    )
    end.set_uniform_lc(lc)

    return [start, cylinder, end, inner_cylinder]

# start, cylinder, end, inner_cylinder = annulus(2, 1, 10, 0.5, -0.1, -0.4, 5e-1)
# geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[1, 2, 3, 4]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)

class FauxGeom():
    def init(self):
        self.gmsh = None
        self.V = None
        self.msh = None
        self.facet_tags = None


def sphere(r, epsilon, lc, show_mesh=False):
    geom = FauxGeom()
    geom.gmsh = gmsh
    geom.gmsh.initialize()
    geom.gmsh.model.remove_physical_groups()
    geom.gmsh.model.remove()

    gmsh.option.set_number("Mesh.MeshSizeFactor", lc)
    circle = geom.gmsh.model.occ.add_sphere(0, 0, 0, r+epsilon)
    geom.gmsh.model.occ.synchronize()
    geom.gmsh.model.add_physical_group(3, [circle], 9)
    geom.gmsh.model.add_physical_group(2, [circle], 1)
    # geom.gmsh.model.add_physical_group(1, [circle_loop], 9)
    geom.gmsh.model.mesh.generate(3)
    geom.msh, _, geom.facet_tags = gmshio.model_to_mesh(geom.gmsh.model, MPI.COMM_WORLD, 0)

    if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()

    return geom

# sphere(1, 0, 1e-1, show_mesh=True)

def smooth_annulus(ro0, ri0, l0, ro_ep, ri_ep, l_ep, lc):
    ro = ro0 + ro_ep
    ri = ri0 + ri_ep
    l = l0 + l_ep
    points = [
        [[2*ro, 0, 0],
        [2*ro, ro, 0],
        [0, ro, 0],
        [-2*ro, ro, 0],
        [-2*ro, 0, 0],
        [-2*ro, -ro, 0],
        [0, -ro, 0],
        [2*ro, -ro, 0],
        [2*ro, 0, 0]],
        [[ro, 0, l],
        [ro, 2*ro, l],
        [0, 2*ro, l],
        [-ro, 2*ro, l],
        [-ro, 0, l],
        [-ro, -2*ro, l],
        [0, -2*ro, l],
        [ro, -2*ro, l],
        [ro, 0, l]]
    ]

    weights = [[1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1], [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]]
    knotsU = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    knotsV = [0, 1]
    multiplicitiesU = [4, 1, 1, 1, 1, 1, 4]
    multiplicitiesV = [2, 2]

    cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 3,
        degreeV = 1
    )
    cylinder.set_uniform_lc(lc)

    dx = (ro - 0.75*ri)*0.75
    dy = (ro - ri) * 0.75
    points = [
        [[1.25*ri + dx, 0 + dy, 0],
        [1.25*ri + dx, ri + dy, 0],
        [0 + dx, ri + dy, 0],
        [-1.25*ri + dx, ri + dy, 0],
        [-1.25*ri + dx, 0 + dy, 0],
        [-1.25*ri + dx, -ri + dy, 0],
        [0 + dx, -ri + dy, 0],
        [1.25*ri+ dx, -ri + dy, 0],
        [1.25*ri + dx, 0 + dy, 0]],
        [[ri + dy, 0 - dx, l],
        [ri + dy, 1.25*ri - dx, l],
        [0 + dy, 1.25*ri - dx, l],
        [-ri + dy, 1.25*ri - dx, l],
        [-ri + dy, 0 - dx, l],
        [-ri + dy, -1.25*ri - dx, l],
        [0 + dy, -1.25*ri - dx, l],
        [ri + dy, -1.25*ri - dx, l],
        [ri + dy, 0 - dx, l]]
    ]
    inner_cylinder = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 3,
        degreeV = 1
    )
    inner_cylinder.set_uniform_lc(lc)
    inner_cylinder.flip_norm = True

    points = [
        [[2*ro, 0, 0],
        [2*ro, ro, 0],
        [0, ro, 0],
        [-2*ro, ro, 0],
        [-2*ro, 0, 0],
        [-2*ro, -ro, 0],
        [0, -ro, 0],
        [2*ro, -ro, 0],
        [2*ro, 0, 0]],
        [[1.25*ri + dx, 0 + dy, 0],
        [1.25*ri + dx, ri + dy, 0],
        [0 + dx, ri + dy, 0],
        [-1.25*ri + dx, ri + dy, 0],
        [-1.25*ri + dx, 0 + dy, 0],
        [-1.25*ri + dx, -ri + dy, 0],
        [0 + dx, -ri + dy, 0],
        [1.25*ri + dx, -ri + dy, 0],
        [1.25*ri + dx, 0 + dy, 0]]
    ]

    start = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 3,
        degreeV = 1
    )
    start.set_uniform_lc(lc)
    start.flip_norm = True

    points = [[
        [ro, 0, l],
        [ro, 2*ro, l],
        [0, 2*ro, l],
        [-ro, 2*ro, l],
        [-ro, 0, l],
        [-ro, -2*ro, l],
        [0, -2*ro, l],
        [ro, -2*ro, l],
        [ro, 0, l]],
        [[ri + dy, 0 - dx, l],
        [ri + dy, 1.25*ri - dx, l],
        [0 + dy, 1.25*ri - dx, l],
        [-ri + dy, 1.25*ri - dx, l],
        [-ri + dy, 0 - dx, l],
        [-ri + dy, -1.25*ri - dx, l],
        [0 + dy, -1.25*ri - dx, l],
        [ri + dy, -1.25*ri - dx, l],
        [ri + dy, 0 - dx, l]]
    ]

    end = NURBsSurface(
        points,
        weights,
        knotsU,
        knotsV,
        multiplicitiesU,
        multiplicitiesV,
        degreeU = 3,
        degreeV = 1
    )
    end.set_uniform_lc(lc)

    return [start, cylinder, end, inner_cylinder]

# start, cylinder, end, inner_cylinder = smooth_annulus(2, 1, 10, 0.5, -0.1, -0.4, 5e-1)
# geom = NURBs3DGeometry([[start, cylinder, end, inner_cylinder]])
# geom.model.remove_physical_groups()
# geom.model.remove()
# geom.generate_mesh()
# geom.add_bspline_groups([[1, 2, 3, 4]])
# geom.model_to_fenics(show_mesh=True)
# geom.create_function_space("Lagrange", 3)