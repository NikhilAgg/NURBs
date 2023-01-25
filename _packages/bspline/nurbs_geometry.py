import gmsh
import sys
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem
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

        #Fenics
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
            for tag, group in zip(self.bspline_tags.flatten(), bspline_groups.flatten()):
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
                multiplicities = bspline.multiplicities
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
        
        self.surface_tag = self.factory.add_plane_surface([*curveloop_tags], 1)

        self.factory.synchronize()
        self.model.mesh.generate(2)

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

        surfaceloop_tag = self.factory.add_surface_loop(surface_tags)

        return [surfaceloop_tag, surface_tags]


    def generate_mesh(self, show_mesh=False):
        surfaceloop_tags = []
        for bspline_list in self.bsplines:
            bspline_tags = self.create_bspline_surfaces(bspline_list)
            
            surfaceloop_tags.append(bspline_tags[0])
            self.bspline_tags.append(bspline_tags)
        
        self.volume_tag = self.factory.add_volume([*surfaceloop_tags], 1)

        self.factory.heal_shapes()
        self.factory.synchronize()
        self.model.mesh.generate(3)

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

