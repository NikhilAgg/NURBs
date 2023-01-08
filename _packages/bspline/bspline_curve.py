import numpy as np
import gmsh
import sys

class BSplineCurve:
    def __init__(self, ctrl_points=[], weights=None, knots=[], multiplicities=None, degree=None, lc=None):
        self.ctrl_points = np.array(ctrl_points)
        self.n = len(ctrl_points)
        self.weights = weights or np.ones(self.n)
        self.knots = knots
        self.multiplicities = multiplicities or np.ones(len(knots))
        self.degree = degree
        self.lc = lc

    def set_uniform_lc(self, lc):
        self.lc = np.ones(self.n) * lc

    def is_closed(self):
        return np.all(self.ctrl_points[0] == self.ctrl_points[-1])

    def get_unit_normal(self):
        if not self.is_closed():
            return -1
        v1 = self.ctrl_points[0] - self.ctrl_points[1]

        normal = np.array([0, 0, 0])
        i = 0
        while np.all(normal == 0):
            v2 = self.ctrl_points[i] - self.ctrl_points[1]
            normal = np.cross(v1, v2)
            i += 1

        return normal/np.linalg.norm(normal)

class BSplineSurface:
    def __init__(self, ctrl_points=[], weights=None, knotsU=[], knotsV=[], multiplicitiesU=None, multiplicitiesV=None, degreeU=None, degreeV=None, lc=None):
        self.ctrl_points = ctrl_points
        self.n = len(ctrl_points)
        self.weights = weights if np.any(weights!=None) else np.ones(self.n)
        self.knotsU = knotsU
        self.knotsV = knotsV
        self.multiplicitiesU = multiplicitiesU or np.ones(len(knotsU))
        self.multiplicitiesV = multiplicitiesV or np.ones(len(knotsV))
        self.degreeU = degreeU
        self.degreeV = degreeV
        self.lc = lc
        self.nU = self.degreeU * (len(knotsU) - 1) + 1
        self.nV = self.degreeV * (len(knotsV) - 1) + 1

    def set_uniform_lc(self, lc):
        self.lc = np.ones(self.n) * lc

    def is_closed(self):
        is_closed = True
        points = self.ctrl_points
        nV = self.nV
        nU = self.nU
        
        for i in range(nV):
            if points[i*nU] != points[(i+1)*nU - 1]:
                is_closed = False
        
        return is_closed


def add_bspline_points(factory, curve, point_dict):
        point_tags = []
        for point, lc_i in zip(curve.ctrl_points, curve.lc):
            if tuple(point) not in point_dict:
                tag = factory.addPoint(*point, lc_i)
                point_dict[tuple(point)] = tag
            else:
                tag = point_dict[tuple(point)]

            point_tags.append(tag)

        return point_tags, point_dict

        
class BSpline2DGeometry:
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
    

class BSpline3DGeometry:
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

    
    def create_bspline_surface(self, bsplines):
        surface_tags = []
        for bspline in bsplines:
            point_tags, self.point_dict = add_bspline_points(self.factory, bspline, self.point_dict)

            surface_tag = self.factory.add_bspline_surface(
                point_tags,
                bspline.nU,
                degreeU = bspline.degreeU,
                degreeV = bspline.degreeV,
                weights = bspline.weights, 
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
        # self.model.mesh.generate(3)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()