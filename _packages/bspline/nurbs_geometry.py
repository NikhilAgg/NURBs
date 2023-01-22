import gmsh
import sys

class NURBs2DGeometry:
    def __init__(self, outer_bsplines=[], inner_bsplines=[]):
        self.outer_bsplines = outer_bsplines
        self.inner_bsplines = inner_bsplines

        self.outer_bspline_tags = []
        self.innner_bspline_tags = []

        self.point_dict = {}

        #Gmsh
        self.gmsh = gmsh
        self.gmsh.initialize()
        self.model = self.gmsh.model
        self.factory = self.model.occ

        #TODO Add checks that bsplines are closed, and in the same plane


    def create_bspline_curve(self, bsplines):
        curve_tags = []
        for bspline in bsplines:
            point_tags, self.point_dict = add_bspline_points(self.factory, bspline, self.point_dict)

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
        self.outer_bspline_tags = self.create_bspline_curve(self.outer_bsplines)
        outer_curveloop_tag = self.outer_bspline_tags[0]
        
        self.innner_bspline_tags = []
        curveloop_tags = []
        for bspline_list in self.inner_bsplines:
            bspline_tags = self.create_bspline_curve(bspline_list)
            
            curveloop_tags.append(bspline_tags[0])
            self.innner_bspline_tags.append(bspline_tags)
        
        self.surface_tag = self.factory.add_plane_surface([outer_curveloop_tag, *curveloop_tags], 1)

        self.factory.synchronize()
        self.model.mesh.generate(2)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()
    

class NURBs3DGeometry:
    def __init__(self, outer_bsplines=[], inner_bsplines=[]):
        self.outer_bsplines = outer_bsplines
        self.inner_bsplines = inner_bsplines

        self.outer_bspline_tags = []
        self.innner_bspline_tags = []

        self.point_dict = {}

        #Gmsh
        self.gmsh = gmsh
        self.gmsh.initialize()
        self.model = self.gmsh.model
        self.factory = self.model.occ

        #TODO Add checks that bsplines are closed and curveloop can be made
        #TODO Fix degenerate bspline case

    
    def create_bspline_surface(self, bsplines):
        surface_tags = []
        for bspline in bsplines:
            point_tags, self.point_dict = add_bspline_points(self.factory, bspline, self.point_dict)

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
        self.outer_bspline_tags = self.create_bspline_surface(self.outer_bsplines)
        outer_surfaceloop_tag = self.outer_bspline_tags[0]
        
        self.innner_bspline_tags = []
        surfaceloop_tags = []
        for bspline_list in self.inner_bsplines:
            bspline_tags = self.create_bspline_curve(bspline_list)
            
            surfaceloop_tags.append(bspline_tags[0])
            self.innner_bspline_tags.append(bspline_tags)
        
        self.volume_tag = self.factory.add_volume([outer_surfaceloop_tag, *surfaceloop_tags], 1)

        self.factory.heal_shapes()
        self.factory.synchronize()
        self.model.mesh.generate(3)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()

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