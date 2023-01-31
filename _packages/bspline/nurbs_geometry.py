import gmsh
import sys
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem, mesh
from scipy.interpolate import LinearNDInterpolator

class NURBsGeometry:
    def __init__(self, bsplines = []):
        self.bsplines = bsplines

        self.bspline_tags = []

        self.bspline_groups = []

        self.point_dict = {}

        #Gmsh
        self.gmsh = gmsh
        self.gmsh.initialize()
        self.model = self.gmsh.model
        self.factory = self.model.occ
        self.node_to_params = {}

        #Fenics
        self.cell_type = None
        self.degree = None
        self.V = None
        self.boundary_conditions = None


    def add_bspline_points(self, curve):
        point_tags = []

        ctrl_points = curve.ctrl_points.reshape(-1, curve.dim)
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
                

    def model_to_fenics(self, comm=MPI.COMM_WORLD, rank=0):
        self.model.mesh.generate(self.dim)
        self.msh, self.cell_markers, self.facet_markers = gmshio.model_to_mesh(self.model, comm, rank)


    def create_function_space(self, cell_type, degree):
        if not self.msh:
            print("Geometry has no fenics msh. Run model_to_fenics to create a msh")
            return

        self.cell_type = cell_type
        self.degree = degree
        self.V = fem.FunctionSpace(self.msh, (cell_type, degree))

        return self.V

    
    def get_displacement_field(self, typ, bspline_ind, param_ind, flip=False):
        V = fem.VectorFunctionSpace(self.msh, (self.cell_type, 1))
        V2 = fem.VectorFunctionSpace(self.msh, (self.cell_type, self.degree))
        c = fem.Function(V)
        C = fem.Function(V2)
        
        deriv_bsplines = self.get_deriv_bsplines(typ, bspline_ind, param_ind)
        def get_displacement(x):
            C = np.zeros((len(x), len(x[0])), dtype=np.complex128)
            for k, coord in enumerate(zip(x[0], x[1], x[2])):
                if tuple(coord) not in self.node_to_params:
                    continue
                
                for indices, params in self.node_to_params[tuple(coord)]:
                    if tuple(bspline_ind) in deriv_bsplines: 
                        i, j = indices
                        param_ind = deriv_bsplines[tuple(bspline_ind)][0]
                        C[0:self.dim][k] += self.bsplines[i][j].get_displacement(typ, *params, *param_ind, flip)

            return C

        c.interpolate(get_displacement)
        C.interpolate(c)

        return C

    
    def get_deriv_bsplines(self, typ, bspline_ind, param_ind):
        if typ != "control point":
            return {tuple(bspline_ind): param_ind}

        deriv_bsplines = {}
        i, j = bspline_ind
        if len(param_ind) == 1:
            point = self.bsplines[i][j].ctrl_points[param_ind[0]]
        elif len(param_ind) == 1:
            point = self.bsplines[i][j].ctrl_points[param_ind[1]]

        for j, bspline in enumerate(self.bsplines[i]):
            if tuple(point) in bspline.ctrl_point_dict:
                param_inds = bspline.ctrl_point_dict[tuple(point)]
                deriv_bsplines[(i, j)] =  param_inds

        return deriv_bsplines


    def create_node_to_param_map(self):
        self.Vu = fem.FunctionSpace(self.msh, (self.cell_type, 1))
        dof_layout = self.Vu.dofmap.dof_layout
        coords = self.Vu.tabulate_dof_coordinates()

        self.msh.topology.create_connectivity(self.msh.topology.dim-1, self.msh.topology.dim)
        boundary_facets = mesh.exterior_facet_indices(self.msh.topology)
        f_to_c = self.msh.topology.connectivity(self.msh.topology.dim-1, self.msh.topology.dim)
        c_to_f = self.msh.topology.connectivity(self.msh.topology.dim, self.msh.topology.dim-1)

        for facet in boundary_facets:
            cells = f_to_c.links(facet)
            facets = c_to_f.links(cells[0])
            local_index = np.flatnonzero(facets == facet)

            closure_dofs = dof_layout.entity_closure_dofs(
                                self.msh.topology.dim-1,  local_index)
            cell_dofs = self.Vu.dofmap.cell_dofs(cells[0])

            for dof in closure_dofs:
                local_dof = cell_dofs[dof]
                dof_coordinate = coords[local_dof]
                self.get_bspline_params_from_coord(dof_coordinate)

    
    def get_bspline_params_from_coord(self, coord):
        for i, row in enumerate(self.bspline_tags):
            for j, tag in enumerate(row[1]):
                if not self.is_coord_on_bspline(coord, tag):
                    continue
                
                params = self.model.getParametrization(self.dim-1, tag, coord)
                self.add_params_to_map(coord, [i, j], params)
                

    def add_params_to_map(self, coord, indices, params):
        if tuple(coord) not in self.node_to_params:
            self.node_to_params[tuple(coord)] = [[indices, params]]
        else:
            for ind, _ in self.node_to_params[tuple(coord)]:
                if indices == ind:
                    return

            self.node_to_params[tuple(coord)].append([indices, params])

                    
    def in_bounding_box(self, coord, box):
        for i in range(self.dim):
            if coord[i] < box[i] or coord[i] > box[i+self.dim]:
                return False
        
        return True


    def is_coord_on_bspline(self, coord, bspline_tag):
        bounding_box = self.model.getBoundingBox(self.dim-1, bspline_tag)
                
        if self.in_bounding_box(coord, bounding_box):
            if self.model.isInside(self.dim-1, bspline_tag, coord):
                return True

        return False


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

            curve_tag = self.factory.add_bspline(
                point_tags,
                -1, 
                degree = bspline.degree,
                weights = bspline.weights,
                knots = bspline.knots,
                multiplicities = bspline.multiplicities,
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

            surface_tag = self.factory.add_bspline_surface(
                point_tags,
                bspline.nU,
                degreeU = bspline.degreeU,
                degreeV = bspline.degreeV,
                weights = bspline.weights.flatten(), 
                knotsU = bspline.knotsU,
                knotsV = bspline.knotsV,
                multiplicitiesU = bspline.multiplicitiesU,
                multiplicitiesV = bspline.multiplicitiesV
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

