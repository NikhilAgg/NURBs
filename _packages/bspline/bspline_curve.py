import numpy as np
import gmsh
import sys
from .generator_utils import *
from .gmsh_utils import add_bspline_points
from itertools import chain

class BSplineCurve:
    def __init__(self, ctrl_points=[], weights=None, knots=[], multiplicities=None, degree=None, lc=None):
        self.ctrl_points = np.array(ctrl_points)
        self.n = len(ctrl_points)
        self.weights = np.array(weights) if weights else np.ones(self.n)
        
        self.knots = knots
        self.multiplicities = multiplicities or np.ones(len(knots))
        self.U = list(chain.from_iterable([[knots[i]] * multiplicities[i] for i in range(len(knots))]))
        
        self.degree = degree
        self.lc = lc

        #TODO FIX U DIFFERENCE AND PERIODIC SPLINES
        #TODO ADD CHECKS
        

    def set_uniform_lc(self, lc):
        self.lc = np.ones(self.n) * lc


    def is_closed(self):
        return np.all(self.ctrl_points[0] == self.ctrl_points[-1])


    def derivative_wrt_ctrl_point(self, i_deriv, u):
        i = find_span(self.n, self.degree, u, self.U)

        if i < i_deriv or i > i_deriv + self.degree:
            return 0
            
        N = nurb_basis(i, self.degree, u, self.U)
        
        denom = 0.0

        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            denom += N[k] * self.weights[ind + k]
        
        return (N[i_deriv - i + self.degree] * self.weights[i_deriv]) / denom


    def calculate_point(self, u):
        P_w = np.column_stack(((self.ctrl_points.T * self.weights).T, self.weights))

        i = find_span(self.n, self.degree, u, self.U)
        N = nurb_basis(i, self.degree, u, self.U)

        curve = np.zeros(len(P_w[0]))

        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            curve += P_w[ind + k] * N[k]

        return curve[:-1]/curve[-1]




class BSplineSurface:
    def __init__(self, ctrl_points=[], weights=None, knotsU=[], knotsV=[], multiplicitiesU=None, multiplicitiesV=None, degreeU=None, degreeV=None, lc=None):
        self.ctrl_points = np.array(ctrl_points)
        self.n = len(ctrl_points)
        self.weights = np.array(weights) if np.any(weights!=None) else np.ones(self.n)

        self.knotsU = knotsU
        self.knotsV = knotsV
        self.multiplicitiesU = multiplicitiesU or np.ones(len(knotsU))
        self.multiplicitiesV = multiplicitiesV or np.ones(len(knotsV))
        self.U = list(chain.from_iterable([[knotsU[i]] * multiplicitiesU[i] for i in range(len(knotsU))]))
        self.V = list(chain.from_iterable([[knotsV[i]] * multiplicitiesV[i] for i in range(len(knotsV))]))

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


    def calculate_point(self, u, v):
        P_w = np.column_stack(((self.ctrl_points.T * self.weights).T, self.weights))
        n_u = len(self.U) - self.degreeU - 1
        n_v = len(self.V) - self.degreeV - 1
        
        i_u = find_span(n_u, self.degreeU, u, self.U)
        i_v = find_span(n_v, self.degreeV, v, self.V)
        
        N_u = nurb_basis(i_u, self.degreeU, u, self.U)
        N_v = nurb_basis(i_v, self.degreeV, v, self.V)

        temp = np.zeros((self.degreeV+1, len(P_w[0])))
        for j in range(self.degreeV + 1):
            ind, no_nonzero_basis = find_inds(i_u, self.degreeU)
            ind += j * n_u
            for k in range(no_nonzero_basis):
                temp[j] += P_w[ind + k] * N_u[k]
                
        surface = np.zeros(len(P_w[0]))
        for l in range(self.degreeV + 1):
            surface += N_v[l] * temp[l]

        return surface[:-1]/surface[-1]


    def derivative_wrt_ctrl_point(self, i_deriv, j_deriv, u, v):
        i_u = find_span(self.nU, self.degreeU, u, self.U)
        i_v = find_span(self.nV, self.degreeV, v, self.V)

        if i_u < i_deriv or i_u > i_deriv + self.degreeU or i_v < j_deriv or i_v > j_deriv + self.degreeV:
            return 0
            
        N_u = nurb_basis(i_u, self.degreeU, u, self.U)
        N_v = nurb_basis(i_v, self.degreeV, v, self.V)
        
        denom = 0.0
        ind_v, _ = find_inds(i_v, self.degreeV)
        ind_u, _ = find_inds(i_u, self.degreeU)
        for k in range(len(N_v)):
            for l in range(len(N_u)):
                denom += N_v[k] * N_u[l] * self.weights[(ind_v + k) * self.nU + (ind_u + l)]
        
        return (N_u[i_deriv - i_u + self.degreeU] * N_v[j_deriv - i_v + self.degreeV] * self.weights[j_deriv * self.nU + i_deriv]) / denom


        
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
        self.model.mesh.generate(3)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()