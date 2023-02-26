import gmsh
import sys
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem, mesh
from scipy.interpolate import LinearNDInterpolator
import ufl

class NURBsGeometry:
    def __init__(self, bsplines = []):
        self.bsplines = bsplines

        self.bspline_tags = []

        self.bspline_groups = []

        self.point_dict = {}

        #Gmsh
        self.gmsh = gmsh
        self.gmsh.initialize() if not self.gmsh.isInitialized() else None
        self.model = self.gmsh.model
        self.factory = self.model.occ
        self.node_to_params = {}

        #Fenics
        self.V = None
        self.boundary_conditions = None


    def add_bspline_points(self, curve):
        point_tags = []

        ctrl_points = curve.ctrl_points.reshape(-1, curve.dim)
        ctrl_points = np.c_[ctrl_points, np.zeros((ctrl_points.shape[0], np.abs(3-self.dim)))] #Makes the ctrl points at least 3D so that they can be passed into gmsh
        for point, lc_i in zip(ctrl_points, curve.lc.flatten()):
            if tuple(point) not in self.point_dict:
                tag = self.factory.addPoint(*point, lc_i)
                self.point_dict[tuple(point)] = tag
            else:
                tag = self.point_dict[tuple(point)]

            point_tags.append(tag)

        return point_tags


    def add_bspline_groups(self, bspline_groups, group_names=None):
        self.bspline_groups = bspline_groups

        if self.bspline_groups != []:
            for i, row in enumerate(bspline_groups):
                for j, group in enumerate(row):
                    self.model.addPhysicalGroup(self.dim-1, [self.bspline_tags[i][1][j]], group)

        if group_names:
            for tag, name in group_names:
                self.model.setPhysicalName(self.dim-1, tag, name)
                

    def model_to_fenics(self, comm=MPI.COMM_WORLD, rank=0, show_mesh=False):
        self.model.mesh.generate(self.dim)
        self.msh, self.cell_markers, self.facet_markers = gmshio.model_to_mesh(self.model, comm, rank)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


    def create_function_space(self, cell_type, degree):
        if not self.msh:
            print("Geometry has no fenics msh. Run model_to_fenics to create a msh")
            return

        self.cell_type = cell_type
        self.degree = degree
        self.V = fem.FunctionSpace(self.msh, (cell_type, degree))

        return self.V


    def get_displacement_field(self, typ, bspline_ind, param_ind, flip=False, degree=None, cell_type=None):
        deriv_bsplines = self.get_deriv_bsplines(typ, bspline_ind, param_ind)

        if typ == "control point":
            dim = self.dim
        else:
            dim = 1
  
        def get_displacement(bsplines, coord):
            nonlocal typ, flip, deriv_bsplines
            for indices, params in bsplines:
                C = np.zeros(dim)
                if tuple(indices) in deriv_bsplines: 
                    i, j = indices
                    param_ind = deriv_bsplines[tuple(indices)][0]
                    C += self.bsplines[i][j].get_displacement(typ, *params, *param_ind, flip)

            return C

        degree = degree or self.degree
        cell_type = cell_type or self.cell_type
        return self.function_from_params(dim, get_displacement, degree, cell_type)

    
    def get_normalised_displacement_field(self, typ, bspline_ind, param_ind, flip=False, degree=None, cell_type=None):
        C = self.get_displacement_field(typ, bspline_ind, param_ind, flip, degree, cell_type)
        C_mag = fem.assemble_scalar(C*ufl.dS)
        return C/C_mag

    
    def function_from_params(self, dim, func, degree=None, cell_type=None):
        degree = degree or self.degree
        cell_type = cell_type or self.cell_type

        if dim == 1:
            V = fem.FunctionSpace(self.msh, (cell_type, 1))
            V2 = fem.FunctionSpace(self.msh, (cell_type, degree))
        else:
            V = fem.VectorFunctionSpace(self.msh, (cell_type, 1), dim=dim)
            V2 = fem.VectorFunctionSpace(self.msh, (cell_type, degree), dim=dim)

        c = fem.Function(V)
        C = fem.Function(V2)

        def interp_func(x):
            nonlocal dim
            C = np.zeros((dim, len(x[0])), dtype=np.complex128)
            i=0
            dictd = {}
            us = []
            for k, coord in enumerate(zip(x[0], x[1], x[2])):
                coord = np.around(coord, self.node_map_decimal_points)
                if tuple(coord) not in self.node_to_params:
                    continue
                i+=1
                if k==251:
                    pass
                if tuple(coord) not in dictd:
                    dictd[tuple(coord)] = [func(self.node_to_params[tuple(coord)], coord), self.node_to_params[tuple(coord)][0][1], k]
                    us.append(self.node_to_params[tuple(coord)][0][1])
                C[:, k] = func(self.node_to_params[tuple(coord)], coord)

            dictd = dict(sorted(dictd.items(), key=lambda x: x[1][1][0]))
            return C

        c.interpolate(interp_func)
        C.interpolate(c)

        return C

    
    def get_deriv_bsplines(self, typ, bspline_ind, param_ind):
        if typ != "control point":
            return {tuple(bspline_ind): [param_ind]}

        deriv_bsplines = {}
        i, j = bspline_ind
        if len(param_ind) == 1:
            point = self.bsplines[i][j].ctrl_points[param_ind[0]]
        elif len(param_ind) == 2:
            point = self.bsplines[i][j].ctrl_points[param_ind[0]][param_ind[1]]

        for j, bspline in enumerate(self.bsplines[i]):
            if bspline.ctrl_points_dict == {}:
                bspline.create_ctrl_point_dict()

            if tuple(point) in bspline.ctrl_points_dict:
                param_inds = bspline.ctrl_points_dict[tuple(point)]
                deriv_bsplines[(i, j)] =  param_inds

        return deriv_bsplines

    
    def create_node_to_param_map(self, decimal_points=8):
        self.node_map_decimal_points = decimal_points
        for i, row in enumerate(self.bspline_tags):
            for j, tag in enumerate(row[1]):
                nodes = self.model.mesh.getNodes(self.dim-1, tag)[1]
                params = self.model.getParametrization(self.dim-1, tag, nodes)
                p = self.dim-1
                for k in range(len(params)//p):
                    self.add_params_to_map(nodes[3*k:3*k+3], [i, j], params[p*k:p*k+p])
                

    def add_params_to_map(self, coord, indices, params):
        coord = np.around(coord, self.node_map_decimal_points)
        if tuple(coord) not in self.node_to_params:
            self.node_to_params[tuple(coord)] = [[indices, params]]
        else:
            for ind, _ in self.node_to_params[tuple(coord)]:
                if indices == ind:
                    return

            self.node_to_params[tuple(coord)].append([indices, params])


    def set_boundary_conditions(self, boundary_conditions):
        self.boundary_conditions = boundary_conditions


class NURBs2DGeometry(NURBsGeometry):
    def __init__(self, bsplines=[]):
        self.dim = 2
        super().__init__(bsplines)
        #TODO Add checks that bsplines are closed, and in the same plane


    def create_bspline_curves(self, bsplines):
        curve_tags = []
        for bspline in bsplines:
            point_tags = self.add_bspline_points(bspline)

            multiplicities = bspline.get_periodic_multiplicities()

            curve_tag = self.factory.add_bspline(
                point_tags,
                -1, 
                degree = bspline.degree,
                weights = bspline.weights,
                knots = bspline.knots,
                multiplicities = multiplicities,
            )

            curve_tags.append(curve_tag)

        curveloop_tag = gmsh.model.occ.addCurveLoop(curve_tags)
            
        return [curveloop_tag, curve_tags]


    def generate_mesh(self, show_mesh=False):
        curveloop_tags = []
        for bspline_list in self.bsplines:
            bspline_tags = self.create_bspline_curves(bspline_list)
            
            curveloop_tags.append(bspline_tags[0])
            self.bspline_tags.append(bspline_tags)
        
        self.surface_tag = self.factory.add_plane_surface([*curveloop_tags])

        self.factory.synchronize()
        self.model.addPhysicalGroup(self.dim, [self.surface_tag], 1)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


class NURBs3DGeometry(NURBsGeometry):
    def __init__(self, bsplines=[]):
        self.dim = 3
        super().__init__(bsplines)

        #Other
        self.ctrl_point_dict = {}

        #TODO Add checks that bsplines are closed and curveloop can be made
        #TODO Fix degenerate bspline case

    
    def create_bspline_surfaces(self, bsplines):
        surface_tags = []
        for bspline in bsplines:
            point_tags = self.add_bspline_points(bspline)

            multiplicitiesU, multiplicitiesV = bspline.get_periodic_multiplicities()
            surface_tag = self.factory.add_bspline_surface(
                point_tags,
                bspline.nU,
                degreeU = bspline.degreeU,
                degreeV = bspline.degreeV,
                weights = bspline.weights.flatten(), 
                knotsU = bspline.knotsU,
                knotsV = bspline.knotsV,
                multiplicitiesU = multiplicitiesU,
                multiplicitiesV = multiplicitiesV
            )
            surface_tags.append(surface_tag)

        self.factory.heal_shapes(makeSolids=False)
        surfaceloop_tag = self.factory.add_surface_loop(surface_tags)

        return [surfaceloop_tag, surface_tags]


    def generate_mesh(self, show_mesh=False):
        surfaceloop_tags = []
        for bspline_list in self.bsplines:
            bspline_tags = self.create_bspline_surfaces(bspline_list)
            
            surfaceloop_tags.append(bspline_tags[0])
            self.bspline_tags.append(bspline_tags)
        
        self.volume_tag = self.factory.add_volume([*surfaceloop_tags], 1)
        self.factory.synchronize()
        self.model.addPhysicalGroup(self.dim, [self.volume_tag], 1)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


    def create_ctrl_point_dict(self):
        for i, bspline_list in enumerate(self.bsplines):
            for j, nurb in enumerate(bspline_list):
                for point in nurb.ctrl_points.reshape(-1, nurb.dim):
                    if tuple(point) in self.ctrl_point_dict:
                        self.ctrl_point_dict[tuple(point)].add((i, j))
                    else:
                        self.ctrl_point_dict[tuple(point)] = set([(i, j)])

