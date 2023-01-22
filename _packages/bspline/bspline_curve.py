import numpy as np
import math
import gmsh
import sys
from .generator_utils import *
from .gmsh_utils import add_bspline_points
from itertools import chain

class BSplineCurve:
    def __init__(self, ctrl_points=[], weights=None, knots=[], multiplicities=None, degree=None, lc=None):
        self.ctrl_points = np.array(ctrl_points)
        self.dim = len(self.ctrl_points[0])
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
            return [0, [0] * self.dim]
            
        N = nurb_basis(i, self.degree, u, self.U)
        
        denom = 0.0
        numerator_sum = 0.0 #This is equal to the summation in the numerator of the second term in the derivative wrt to the weights
        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            common = N[k] * self.weights[ind + k]
            denom += common
            numerator_sum += common * self.ctrl_points[ind + k]

        deriv_wrt_point = (N[i_deriv - i + self.degree] * self.weights[i_deriv]) / denom

        deriv_wrt_weight = (N[i_deriv - i + self.degree] * self.ctrl_points[i_deriv]) / denom - \
                           (N[i_deriv - i + self.degree] * numerator_sum)/denom**2
        
        return [deriv_wrt_point, deriv_wrt_weight]


    def calculate_point(self, u):
        P_w = np.column_stack(((self.ctrl_points.T * self.weights).T, self.weights))

        i = find_span(self.n, self.degree, u, self.U)
        N = nurb_basis(i, self.degree, u, self.U)

        curve = np.zeros(len(P_w[0]))

        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            curve += P_w[ind + k] * N[k]

        return curve[:-1]/curve[-1]


    def derivative_wrt_u(self, u, k):
        A_ders = np.zeros((k+1, self.dim))
        w_ders = np.zeros(k+1)
        ders = np.zeros((k+1, self.dim))

        i = find_span(self.n, self.degree, u, self.U)
        N_ders = nurb_basis_derivatives(i, u, self.degree, self.U, k)
        for l in range(min(k, self.degree) + 1):
            for j in range(self.degree + 1): #Requires clamped start
                temp = N_ders[l][j] * self.weights[i - self.degree + j]
                w_ders[l] += temp
                A_ders[l] += temp * self.ctrl_points[i - self.degree + j]

        for l in range(k + 1):
            v = A_ders[l]
            for i in range(1, l+1):
                v -= math.comb(l, i) * w_ders[i] * ders[l - i]
            ders[l] = v/w_ders[0]

        return ders


    def derivative_wrt_knot(self, n_deriv, u):
        i = find_span(self.n, self.degree, u, self.U)

        i_deriv_left = sum(self.multiplicities[0:n_deriv])-1
        i_deriv_right = i_deriv_left + self.multiplicities[n_deriv]
        if i < i_deriv_right - self.degree - 1 or i > i_deriv_right + self.degree:
            return np.zeros(self.dim)

        N = nurb_basis(i, self.degree, u, self.U)

        U_bar = self.U.copy()
        U_bar.insert(i_deriv_right, self.U[i_deriv_right]) #A copy of the knot vector but with the i_deriv th knot repeated one more time

        if u > self.U[i_deriv_right]:
            N_bar = nurb_basis(i+1, self.degree, u, U_bar)
            N_bar.insert(0, 0)
        else:
            N_bar = nurb_basis(i, self.degree, u, U_bar)

        N_bar.append(0)

        N_deriv_right = []
        ind, no_nonzero_basis = find_inds(i, self.degree)
        for j in range(len(N)):
            if j + ind < i_deriv_right - self.degree - 1 or j + ind > i_deriv_right:
                N_deriv_right.append(0)
            elif j + ind == i_deriv_right - self.degree - 1:
                first_term = N_bar[j+1] / (self.U[j + ind + self.degree + 1] - self.U[j + ind + 1])
                N_deriv_right.append(first_term)
            elif j + ind == i_deriv_right:
                second_term = N_bar[j] / (self.U[j + i] - self.U[j + i - self.degree])
                N_deriv_right.append(-second_term)
            else:
                first_term = N_bar[j+1] / (self.U[j + ind + self.degree + 1] - self.U[j + ind + 1])
                second_term = N_bar[j] / (self.U[j + ind + self.degree] - self.U[j + ind])
                N_deriv_right.append(first_term - second_term)

        denom = 0.0
        first_numerator = 0.0 #This is equal to the summation in the numerator of the first term in the derivative wrt to the knots
        second_numerator = 0.0
        second_numerator_sum = 0.0 
        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            denom += N[k] * self.weights[ind + k]
            first_numerator += N_deriv_right[k] * self.weights[ind + k] * self.ctrl_points[ind + k]
            second_numerator += N[k] * self.weights[ind + k] * self.ctrl_points[ind + k]
            second_numerator_sum += N_deriv_right[k] * self.weights[ind + k]

        deriv = first_numerator / denom - (second_numerator * second_numerator_sum) / denom**2

        return deriv
                

    def plot_nurb(self, u, i, U):
        j = find_span(self.n, self.degree, u, U)

        N = nurb_basis(j, self.degree, u, U)

        if u > 0.5:
            pass

        ind, _ = find_inds(j, self.degree)
        for k in range(len(N)):
            if ind + k == i:
                return N[k]
        
        return 0



class BSplineSurface:
    def __init__(self, ctrl_points=[], weights=None, knotsU=[], knotsV=[], multiplicitiesU=None, multiplicitiesV=None, degreeU=None, degreeV=None, lc=None):
        self.ctrl_points = np.array(ctrl_points)
        self.weights = np.array(weights) if np.any(weights!=None) else np.ones(self.ctrl_points.shape)
        self.nV = self.ctrl_points.shape[0]
        self.nU = self.ctrl_points.shape[1]
        self.n = self.nU * self.nV
        self.dim = len(self.ctrl_points[0][0])
        
        self.knotsU = knotsU
        self.knotsV = knotsV
        self.multiplicitiesU = multiplicitiesU or np.ones(len(knotsU))
        self.multiplicitiesV = multiplicitiesV or np.ones(len(knotsV))
        self.U = list(chain.from_iterable([[knotsU[i]] * multiplicitiesU[i] for i in range(len(knotsU))]))
        self.V = list(chain.from_iterable([[knotsV[i]] * multiplicitiesV[i] for i in range(len(knotsV))]))

        self.degreeU = degreeU
        self.degreeV = degreeV
        self.lc = lc
        

    def set_uniform_lc(self, lc):
        self.lc = np.ones(self.ctrl_points.shape) * lc


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
        i_u = find_span(self.nU, self.degreeU, u, self.U)
        i_v = find_span(self.nV, self.degreeV, v, self.V)
        
        N_u = nurb_basis(i_u, self.degreeU, u, self.U)
        N_v = nurb_basis(i_v, self.degreeV, v, self.V)

        temp = np.zeros((self.degreeV+1, self.dim+1))
        for j in range(self.degreeV + 1):
            ind, no_nonzero_basis = find_inds(i_u, self.degreeU)
            for k in range(no_nonzero_basis):
                common = self.weights[j][ind + k] * N_u[k]
                temp[j][-1] += common
                temp[j][0:-1] += self.ctrl_points[j][ind + k] * common
                
        surface = np.zeros(self.dim+1)
        for l in range(self.degreeV + 1):
            surface += N_v[l] * temp[l]

        return surface[:-1]/surface[-1]


    def derivative_wrt_ctrl_point(self, i_deriv, j_deriv, u, v):
        i_u = find_span(self.nU, self.degreeU, u, self.U)
        i_v = find_span(self.nV, self.degreeV, v, self.V)

        if i_u < i_deriv or i_u > i_deriv + self.degreeU or i_v < j_deriv or i_v > j_deriv + self.degreeV:
            return [0, [0, 0, 0]]
            
        N_u = nurb_basis(i_u, self.degreeU, u, self.U)
        N_v = nurb_basis(i_v, self.degreeV, v, self.V)
        
        denom = 0.0
        numerator_sum = 0.0 ##This is equal to the summation in the numerator of the second term in the derivative wrt to the weights
        ind_v, _ = find_inds(i_v, self.degreeV)
        ind_u, _ = find_inds(i_u, self.degreeU)
        for k in range(len(N_v)):
            for l in range(len(N_u)):
                common = N_v[k] * N_u[l] * self.weights[ind_v + k][ind_u + l]
                denom += common
                numerator_sum += common * self.ctrl_points[ind_v + k][ind_u + l]

        deriv_wrt_point = (N_u[i_deriv - i_u + self.degreeU] * N_v[j_deriv - i_v + self.degreeV] * self.weights[j_deriv][i_deriv]) / denom
        
        deriv_wrt_weight = (N_u[i_deriv - i_u + self.degreeU] * N_v[j_deriv - i_v + self.degreeV] * self.ctrl_points[j_deriv][i_deriv]) / denom - \
                            (numerator_sum * N_u[i_deriv - i_u + self.degreeU] * N_v[j_deriv - i_v + self.degreeV])/denom**2
        
        return [deriv_wrt_point, deriv_wrt_weight]

    
    def derivative_wrt_uv(self, u, v, k):
        A_ders, w_ders = self.calculate_num_and_den_derivatives(u, v, k)
        ders = np.zeros((k+1, k+1, self.dim))

        for l in range(k+1):
            for m in range(k-l+1):
                v = A_ders[l][m]
                for j in range(1, m+1):
                    v -= math.comb(m, j) * w_ders[0][j] * ders[l][m-j]
                
                for i in range(1, l+1):
                    v -= math.comb(l, i) * w_ders[i][0] * ders[l-i][m]
                    v2 = 0.0
                    for j in range(1, m+1):
                        v2 += math.comb(m,j) * w_ders[i][j] * ders[l-i][m-j]
                    v -= math.comb(l, i) * v2

                ders[l][m] = v/w_ders[0][0]

        return ders


    def calculate_num_and_den_derivatives(self, u, v, k):
        A_ders = np.zeros((k+1, k+1, self.dim))
        w_ders = np.zeros((k+1, k+1))

        i_u = find_span(self.nU, self.degreeU, u, self.U)
        N_ders_u = nurb_basis_derivatives(i_u, u, self.degreeU, self.U, k)

        i_v = find_span(self.nV, self.degreeV, v, self.V)
        N_ders_v = nurb_basis_derivatives(i_v, v, self.degreeV, self.V, k)

        temp_A = np.zeros((self.degreeV+1, self.dim))
        temp_w = np.zeros(self.degreeV+1)
        for l in range(min(k, self.degreeU) + 1):
            for s in range(self.degreeV + 1):
                temp_w[s] = 0.0
                temp_A[s] = [0.0] * self.dim
                for r in range(self.degreeU + 1):
                    common = N_ders_u[l][r] * self.weights[(i_v-self.degreeV+s) * self.nU + (i_u-self.degreeU+r)]
                    temp_w[s] += common
                    temp_A[s] += common * self.ctrl_points[(i_v-self.degreeV+s) * self.nU + (i_u-self.degreeU+r)]

            for m in range(min(k-l, min(k, self.degreeV))+1):
                A_ders[l][m] = [0.0] * self.dim
                w_ders[l][m] = 0.0
                for s in range(self.degreeV+1):
                    A_ders[l][m] += N_ders_v[m][s]*temp_A[s]
                    w_ders[l][m] += N_ders_v[m][s]*temp_w[s]

        return A_ders, w_ders

        

        
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