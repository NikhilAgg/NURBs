import gmsh
import sys
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem

class NURBsGeometry:
    def __init__(self, outer_bsplines=[], inner_bsplines=[]):
        self.outer_bsplines = outer_bsplines
        self.inner_bsplines = inner_bsplines

        self.outer_bspline_tags = []
        self.innner_bspline_tags = []

        self.outer_bspline_groups = []
        self.innner_bspline_groups = []

        self.point_dict = {}

        #Gmsh
        self.gmsh = gmsh
        self.gmsh.initialize()
        self.model = self.gmsh.model
        self.factory = self.model.occ

        #Fenics
        self.V = None
        self.boundary_conditions = None


    def add_bspline_points(factory, curve, point_dict):
        point_tags = []

        ctrl_points = curve.ctrl_points.reshape(-1, curve.dim)
        for point, lc_i in zip(ctrl_points, curve.lc.flatten()):
            if tuple(point) not in point_dict:
                tag = factory.addPoint(*point, lc_i)
                point_dict[tuple(point)] = tag
            else:
                tag = point_dict[tuple(point)]

            point_tags.append(tag)

        return point_tags, point_dict


    def add_bspline_groups(self, outer_bspline_groups, inner_bspline_groups, group_names=None):
        self.outer_bspline_groups = outer_bspline_groups
        self.innner_bspline_groups = inner_bspline_groups

        if self.outer_bspline_tags != []:
            for tag, group in zip(self.outer_bspline_tags, outer_bspline_groups):
                self.model.addPhysicalGroup(self.dim-1, [tag], group)

        if self.inner_bspline_groups != []:
            for tag, group in zip(self.innner_bspline_tags.flatten(), inner_bspline_groups.flatten()):
                self.model.addPhysicalGroup(self.dim-1, [tag], group)

        if group_names:
            for tag, name in group_names:
                self.model.setPhysicalName(self.dim-1, tag, name)
                

    def model_to_fenics(self, comm=MPI.COMM_WORLD, rank=0):
        self.msh, self.cell_markers, self.facet_markers = gmshio.model_to_mesh(self.model, comm, rank, self.dim)


    def create_function_space(self, cell_type, degree):
        if not self.msh:
            print("Geometry has no fenics msh. Run model_to_fenics to create a msh")
            return

        self.V = fem.FunctionSpace(self.msh, (cell_type, degree))

        return self.V


    def set_boundary_conditions(self, boundary_conditions):
        self.boundary_conditions = boundary_conditions


class NURBs2DGeometry(NURBsGeometry):
    def __init__(self, outer_bsplines=[], inner_bsplines=[]):
        self.dim = 2
        super().__init__(outer_bsplines, inner_bsplines)
        #TODO Add checks that bsplines are closed, and in the same plane


    def create_bspline_curves(self, bsplines):
        curve_tags = []
        for bspline in bsplines:
            point_tags, self.point_dict = self.add_bspline_points(self.factory, bspline, self.point_dict)

            curve_tag = self.factory.add_bspline(
                point_tags,
                -1, 
                degree = bspline.degree,
                weights = bspline.weights,
                knots = bspline.knots,
                multiplicities = bspline.multiplicities
            )

            curve_tags.append(curve_tag)

        curveloop_tag = gmsh.model.occ.addCurveLoop(curve_tags)
            
        return [curveloop_tag, curve_tags]


    def generate_mesh(self, show_mesh=False):
        self.outer_bspline_tags = self.create_bspline_curves(self.outer_bsplines)
        outer_curveloop_tag = self.outer_bspline_tags[0]
        
        self.innner_bspline_tags = []
        curveloop_tags = []
        for bspline_list in self.inner_bsplines:
            bspline_tags = self.create_bspline_curves(bspline_list)
            
            curveloop_tags.append(bspline_tags[0])
            self.innner_bspline_tags.append(bspline_tags)
        
        self.surface_tag = self.factory.add_plane_surface([outer_curveloop_tag, *curveloop_tags], 1)

        self.factory.synchronize()
        self.model.mesh.generate(2)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


class NURBs3DGeometry(NURBsGeometry):
    def __init__(self, outer_bsplines=[], inner_bsplines=[]):
        self.dim = 3
        super().__init__(outer_bsplines, inner_bsplines)

        #Other
        self.ctrl_point_dict = {}

        #TODO Add checks that bsplines are closed and curveloop can be made
        #TODO Fix degenerate bspline case

    
    def create_bspline_surfaces(self, bsplines):
        surface_tags = []
        for bspline in bsplines:
            point_tags, self.point_dict = self.add_bspline_points(self.factory, bspline, self.point_dict)

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

        surfaceloop_tag = self.factory.add_surface_loop(surface_tags)

        return surfaceloop_tag, surface_tags


    def generate_mesh(self, show_mesh=False):
        self.outer_bspline_tags = self.create_bspline_surfaces(self.outer_bsplines)
        outer_surfaceloop_tag = self.outer_bspline_tags[0]
        
        self.innner_bspline_tags = []
        surfaceloop_tags = []
        for bspline_list in self.inner_bsplines:
            bspline_tags = self.create_bspline_surfaces(bspline_list)
            
            surfaceloop_tags.append(bspline_tags[0])
            self.innner_bspline_tags.append(bspline_tags)
        
        self.volume_tag = self.factory.add_volume([outer_surfaceloop_tag, *surfaceloop_tags], 1)

        self.factory.heal_shapes()
        self.factory.synchronize()
        self.model.mesh.generate(3)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()

