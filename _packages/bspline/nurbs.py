"""A module that contains classes and helper functions that allow for the definintion of Nurbs curves and surfaces. The module includes
functions for calculting points along the curves, derviatives, displacement fields, etc"""

import numpy as np
import math
from itertools import chain

class NURBsCurve:
    def __init__(self, ctrl_points=[], weights=None, knots=[], multiplicities=None, degree=None, lc=None):
        """
        Sets the NURBs Parameters to those passed in as arguements. 

        Parameters
        ----------------------------
        ctrl_points: list of lists or 2d numpy array
            A list of the control points for the curve. 
        weights: list or 1d numpy array
            A list of weights corresponding to each of the control points defined in the arguement ctrl_points. If none are 
            given, the weights are all set to 1
        knots: list
            A list of the unique knot/node values of the curve
        multiplicities: list of ints
            A list of containing the number of times each knot is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline
        degree: int
            The degree of the bspline curve
        lc: list or 1d numpy array
            A list of floats which correspond to the chararcteristic mesh size at each control point in Gmsh. If set to none, set to 1 for all control points
            (Default: None)
        """

        #Default values
        self.weights = np.ones(len(ctrl_points))
        self.lc = np.ones(len(ctrl_points))
        self.multiplicities = list(np.ones(len(knots)))

        self.set_ctrl_points(ctrl_points, weights, lc)
        self.set_knots(knots, multiplicities)
        self.set_degree(degree)

        #Other
        self.create_ctrl_point_dict()
        self.flip_norm = False

        #TODO FIX U DIFFERENCE AND PERIODIC SPLINES
        #TODO ADD CHECKS


    def set_ctrl_points(self, ctrl_points=None, weights=None, lc=None):
        """
        Set ctrl_points and weights for curve.

        Parameters
        ----------------------------
        ctrl_points: list of lists or 2d numpy array
            A list of the control points for the curve. If set to None, ctrl_points remain unchanged (Default: None)
        weights: list or 1d numpy array
            A list of weights corresponding to each of the control points defined in the arguement ctrl_points. 
            If set to None, ctrl_points remain unchanged (Default: None)
        lc: list or 1d numpy array
            A list of floats which correspond to the chararcteristic mesh size at each control point in Gmsh. If set to none, lc will remain unchanged
            (Default: None)
        """
        self.ctrl_points = np.array(ctrl_points) if np.any(ctrl_points) else self.ctrl_points
        self.dim = len(self.ctrl_points[0])
        self.n = len(ctrl_points)
        self.weights = np.array(weights) if np.any(weights) else self.weights
        self.lc = np.array(lc) if np.any(lc) else self.lc

        assert self.ctrl_points.shape[0] == self.weights.shape[0] and self.ctrl_points.shape[0] == self.lc.shape[0]

    
    def set_knots(self, knots=None, multiplicities=None):
        """
        Set knots for curve.

        Parameters
        ----------------------------
        knots: list
            A list of the unique knot/node values of the curve. If set to None, ctrl_points remain unchanged (Default: None)
        multiplicities: list of ints
            A list of containing the number of times each knot is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline. 
            If set to None, ctrl_points remain unchanged (Default: None)
        """
        self.knots = list(knots) if np.any(knots) else self.knots
        self.multiplicities = list(multiplicities) or self.multiplicities

        assert len(self.knots) == len(self.multiplicities)

        self.U = list(chain.from_iterable([[knots[i]] * multiplicities[i] for i in range(len(knots))]))

    
    def set_degree(self, degree):
        """
        Set degree for curve.

        Parameters
        ----------------------------
        degree: int
            The degree of the bspline curve
        """
        self.degree = degree
        

    def set_uniform_lc(self, lc):
        """
        Sets lc for each control point to the value passed in. Lc defines the element size near each control point when the 
        curve is used to defined a mesh using a NURBsGeometry Object. NOTE: Current versions of gmsh don't support using different
        element sizes for different control points - as a result the lc of the first element is the only one that will be used.

        Parameters
        ----------------------------
        lc: float
            Element size
        """
        self.lc = np.ones(self.n) * lc


    def is_periodic(self):
        """
        Returns whether or not the curve is periodic. This is true if the first and last control points are equal. Otherwise it is false

        Returns
        ----------------------------
        is_periodic: bool
            True if the curve is periodic, False otherwise
        """
        return np.all(self.ctrl_points[0] == self.ctrl_points[-1])
    

    def is_completely_periodic(self):
        """
        Returns whether or not the curve is completely periodic. This is true if the first and last control points are equal and 
        if the first and last weights are equal. Otherwise it is false

        Returns
        ----------------------------
        is_completely_periodic: bool
            True if the curve is completely periodic, False otherwise
        """
        if np.all(self.ctrl_points[0] == self.ctrl_points[-1]) and self.weights[0] == self.weights[-1]:
            return True
        else:
            return False

    
    def get_periodic_multiplicities(self):
        """
        If the curve is periodic (the first and last control points are the same), then this function returns the periodic 
        multiplicities - aka it removes 1 from both the first and last elements in the multiplicities list. Otherwise it just returns
        the multiplicities of the curve

        Returns
        ----------------------------
        multiplicities: list of int
            If the curve is periodic returns the periodic multiplicities otherwise returns the unaltered multiplicities of the curve
        """

        multiplicities = self.multiplicities[:]
        if self.is_periodic():
            multiplicities[0] -= 1
            multiplicities[-1] -= 1

        return multiplicities


    def calculate_point(self, u):
        """
        Returns the coordinate of the point at the given value of u.

        Parameters
        ----------------------------
        u: float
            Value of the parameter u for which the point will be calculated.

        Returns
        ----------------------------
        coord: 1d np.array
            The coordinates of the point. If the u value given is less than the first knot or greater than the last
            the coordinate 0 is returned.
        """

        P_w = np.column_stack(((self.ctrl_points.T * self.weights).T, self.weights))

        i = find_span(self.n, self.degree, u, self.U)
        if i == -1:
             return np.zeros(self.dim)
        
        N = nurb_basis(i, self.degree, u, self.U)

        curve = np.zeros(len(P_w[0]))

        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            curve += P_w[ind + k] * N[k]

        return curve[:-1]/curve[-1]


    def derivative_wrt_ctrl_point(self, i_deriv, u):
        """
        Calculated the derivative of the curve with respect to a control point

        Parameters
        ----------------------------
        i_deriv: int
            The index of the control point for which you want to calculate the derivative
        u: float
            Value of the parameter u for which at which the derivative will be calculated.

        Returns
        ----------------------------
        list [float, 1d numpy array]:
            A list containing the derivative of the curve with repect to the i_deriv th control point in the first element and the derivative of the 
            curve with respect to the i_deriv th weight in the second element. The derivative of the curve at a point in general is a vector describing
            the direction of movement at that point. A float is returned for the derivative of wrt to the control point where the derivative is equal to the float
            multiplied by the 1 vector.
        """

        i = find_span(self.n, self.degree, u, self.U)

        if i < i_deriv or i > i_deriv + self.degree:
            return [0, np.zeros(self.dim)]
            
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


    def derivative_wrt_u(self, u, k):
        """
        Calculates the derivative of the curve with respect to the parameter u

        Parameters
        ----------------------------
        u: float
            Value of the parameter u for which at which the derivative will be calculated.
        k: int
            The order of the derivative you want to calculate

        Returns
        ----------------------------
        ders: np.array
            A numpy array where each row corresponds to the ith derivative wrt u at the given point u. The last row will contain the kth derivative.
        """
         
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

    
    def get_unit_normal(self, u, flip_norm=None):
        """
        Calculates the unit normal to the curve at a given point

        Parameters
        ----------------------------
        u: float
            Value of the parameter u at which the normal will be found
        flip_norm: bool
            If true will return the negative of the calculated normal. If none is passed the value in self.flip_norm will be used (default: None)

        Returns
        ----------------------------
        unit_norm: np.array
            The unit normal
        """
        flip_norm = flip_norm if flip_norm != None else self.flip_norm

        dC = self.derivative_wrt_u(u, 1)[1]
        dC_hat = dC / np.linalg.norm(dC)

        temp = dC_hat[0]
        dC_hat[0] = -dC_hat[1]
        dC_hat[1] = temp

        if flip_norm:
            dC_hat *= -1

        return dC_hat

    
    def get_curvature(self, u, flip_norm=False):
        """
        Calculates the curvature of the curve with respect to the parameter u. The curvature is defined as the divegence of the 
        unit normal along the curve

        Parameters
        ----------------------------
        u: float
            Value of the parameter u for which at which the derivative will be calculated.
        flip_norm: False
            Multiplies the result by -1. Equivalent to considering the unit normal in the opposite direction to the one calculated. If none is passed
            the value in self.flip_norm will be used (default: None)

        Returns
        ----------------------------
        curvature: float
            The calculated curvature
        """

        ders = self.derivative_wrt_u(u, 2)
        dC = ders[1]
        d2C = ders[2]

        if self.dim == 2:
            curvature = np.cross(d2C, dC) / np.linalg.norm(dC)**3
        else:
            curvature = np.linalg.norm(np.cross(d2C, dC)) / np.linalg.norm(dC)**3

        flip_norm = flip_norm if flip_norm != None else self.flip_norm
        if flip_norm:
            curvature *= -1

        return curvature


    def get_displacement(self, typ, u, i_deriv, flip_norm=None, tie=True):
        """
        Calculates the displacement/derivative of a point in the direction normal to the curve wrt a specified parameter

        Parameters
        ----------------------------
        typ: string
            Either "control point" or "weight" depending on which parameter the displacement field will be calculated wrt.
        u: float
            Value of the parameter u at the point at which at which the displacement will be calculated.
        i_deriv: int
            The index of the parameter for which you want to calculate the derivative
        flip_norm: bool
            If true will consider the displacement in the direction of the opposite unit normal. Note that this is equivalent
            to multiplying the result by -1. If None is passed in, the value of flip in self.flip_norm will be used (default: None)
        tie: bool
            if true and the NURB is periodic changes to the first and last parameter will be added to find the displacement field if the 
            points were to both change. If typ == "weight", the first and last weight must also be equal in addition to the curve being periodic.
            (default: True)

        Returns
        ----------------------------
        displacement: np.array or float
            The displacement of the point. If typ == "control point" then a numpy array is returned where each element is the displacement of the point
            with respect to the kth coordinate. If typ == "weight", the displacement is a float.
        """
        u = self.validate_params(u)
        unit_norm = self.get_unit_normal(u, flip_norm=flip_norm)

        if typ == "control point":
            der = self.derivative_wrt_ctrl_point(i_deriv, u)[0]

            if tie and self.is_periodic() and (i_deriv == 0 or i_deriv == self.n-1):
                der += self.derivative_wrt_ctrl_point(self.n - i_deriv - 1, u)[0]   #self.n - i_deriv - 1 gives self.n -1 if i_deriv is 0 and 0 if _deriv if self.n-1

            return unit_norm * der
            
        elif typ == "weight":
            der = self.derivative_wrt_ctrl_point(i_deriv, u)[1]

            if tie and self.is_completely_periodic() and (i_deriv == 0 or i_deriv == self.n-1):
                der += self.derivative_wrt_ctrl_point(self.n - i_deriv - 1, u)[1]   #self.n - i_deriv - 1 gives self.n -1 if i_deriv is 0 and 0 if _deriv if self.n-1

            return np.dot(der, unit_norm)

        elif typ == "knot":
            raise NameError("Not implemented")

        else:
            raise NameError(f"No param called {typ}")


    def create_ctrl_point_dict(self):
        """
        Creates a dictionary mapping each control point coordinate to its indicies. The dictionary is stored in self.ctrl_points_dict. Keys for the
        dictionary are the coordiates as a tuple and the values are a list of indicies
        """
        self.ctrl_points_dict = {}
        for i, point in enumerate(self.ctrl_points):
                if tuple(point) in self.ctrl_points_dict:
                    self.ctrl_points_dict[tuple(point)].append([i])
                else:
                    self.ctrl_points_dict[tuple(point)] = [[i]]


    def validate_params(self, u):
        if np.isclose(u, self.knots[-1]):
            u = self.knots[-1]

        return u
    

    def initialise_nurb(self, ctrl_points=[], weights=None, knots=[], multiplicities=None, degree=None):
        """
        Sets the NURBs Parameters to those passed in as arguements. 

        Parameters
        ----------------------------
        ctrl_points: list of lists or 2d numpy array
            A list of the control points for the curve. 
        weights: list or 1d numpy array
            A list of weights corresponding to each of the control points defined in the arguement ctrl_points. If none are 
            given, the weights are all set to 1
        knots: list
            A list of the unique knot/node values of the curve
        multiplicities: list of ints
            A list of containing the number of times each knot is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline
        degree: int
            The degree of the bspline curve
        """

        self.ctrl_points = np.array(ctrl_points)
        self.dim = len(self.ctrl_points[0])
        self.n = len(ctrl_points)
        self.weights = np.array(weights) if np.any(weights) else np.ones(self.n)
        
        self.knots = list(knots)
        self.multiplicities = list(multiplicities) or list(np.ones(len(knots)))
        self.U = list(chain.from_iterable([[knots[i]] * multiplicities[i] for i in range(len(knots))]))
        
        self.degree = degree


    def knot_insertion(self, u, r, overwrite=False):
        """
        Calculates a new set of control points, weights, knots and multiplicities if the given knot u is inserted into
        the curve r times with no change the shape of the curve.

        Parameters
        ----------------------------
        u: float
            Value of the knot you want to insert
        r: int
            Number of times you want to insert the knot
        overwrite: bool
            If set to true the parameters of the NURBs curve/object will be set to the calculated new set of parameters

        Returns
        ----------------------------
        U: list of floats
            The modified knot vector.
        ctrl_points: 2d numpy array
            The modified control points. 
        weights: 1d numpy array
            The modified weights
        """
        Pw = np.column_stack(((self.ctrl_points.T * self.weights).T, self.weights))

        k = find_span(self.n, self.degree, u, self.U)
        s = 0 if self.U[k] != u else self.multiplicities[self.knots.index(u)]
        mp = self.n + self.degree + 1
        nq = self.n+r
        mq = nq + self.degree + 1
        
        #Create New Knot vector
        uq = [0] * mq
        uq[0 : k+1] = self.U[0 : k+1]
        uq[k+1 : k+r+1] = [u] * r
        uq[k+r+1 : mp+r+1] = self.U[k+1 : mp+1]

        #Create New control points and insert unchanged ones
        Qw = np.zeros((nq, self.dim+1))
        Qw[0 : k-self.degree+1] = Pw[0 : k-self.degree+1]
        Qw[k-s+r : self.n + r + 1] = Pw[k-s : self.n+1]

        #Calculate new control points
        R = np.zeros((self.degree+1, self.dim+1))
        R[0 : self.degree-s+1] = Pw[k-self.degree : k-s+1]
        for j in range(1, r+1):
            L = k - self.degree + j
            for i in range(self.degree - j - s + 1):
                alpha = (u - self.U[L+i])/(self.U[i+k+1]-self.U[L+i])
                R[i] = alpha*R[i+1] + (1.0-alpha)*R[i]
                
            Qw[L] = R[0]
            Qw[k+r-j-s] = R[self.degree-j-s]

        Qw[L+1 : k-s] = R[1:k-s-L]

        weights = Qw[:, -1]
        ctrl_points = (Qw[:, 0:-1].T / weights).T
        knots, multiplicities = U_vec_to_knots(uq)

        if overwrite:
            self.initialise_nurb(ctrl_points, weights, knots, multiplicities, self.degree)

        return uq, ctrl_points, weights
    



class NURBsSurface:
    def __init__(self, ctrl_points=[], weights=None, knotsU=[], knotsV=[], multiplicitiesU=None, multiplicitiesV=None, degreeU=None, degreeV=None, lc=None):
        """
        Sets the NURBs Parameters to those passed in as arguements. 

        Parameters
        ----------------------------
        ctrl_points: list of lists or 3d numpy array
            A list of the control points for the curve. 
        weights: list of lists or 2d numpy array
            An array of weights corresponding to each of the control points defined in the arguement ctrl_points. If none are 
            given, the weights are all set to 1
        knotsU: list
            A list of the unique knot/node values of the curve along lines of constant V
        knotsV: list
            A list of the unique knot/node values of the curve along lines of constant U
        multiplicitiesU: list of ints
            A list of containing the number of times each knotU is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline
        multiplicitiesV: list of ints
            A list of containing the number of times each knotV is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline
        degreeU: int
            The degree of the bspline curve in the U direction
        degreeV: int
            The degree of the bspline curve in the V direction

        """

        self.ctrl_points = np.array(ctrl_points)
        self.weights = np.ones(self.ctrl_points.shape[:2])
        self.lc = np.ones(self.ctrl_points.shape[:2])
        self.multiplicitiesU = list(np.ones(len(knotsU)))
        self.multiplicitiesV = list(np.ones(len(knotsV)))

        self.set_ctrl_points(weights=weights, lc=lc)
        self.set_knots(knotsU, multiplicitiesU, knotsV, multiplicitiesV)
        self.set_degree(degreeU, degreeV)

        #Other
        self.ctrl_points_dict = {}
        self.flip_norm = False
        

    def set_ctrl_points(self, ctrl_points=None, weights=None, lc=None):
        """
        Set ctrl_points and weights for curve.

        Parameters
        ----------------------------
        ctrl_points: list of lists or 2d numpy array
            A list of the control points for the curve. 
        weights: list or 1d numpy array
            A list of weights corresponding to each of the control points defined in the arguement ctrl_points. If none are 
            given, the weights are all set to 1
        lc: list or 1d numpy array
            A list of floats which correspond to the chararcteristic mesh size at each control point in Gmsh. If set to none, lc will remain unchanged
            (Default: None)
        """
        self.ctrl_points = np.array(ctrl_points) if np.any(ctrl_points) else self.ctrl_points
        self.weights = np.array(weights) if np.any(weights!=None) else self.weights
        self.nV = self.ctrl_points.shape[0]
        self.nU = self.ctrl_points.shape[1]
        self.n = self.nU * self.nV
        self.dim = len(self.ctrl_points[0][0])

        assert self.ctrl_points.shape[:2] == self.weights.shape[:2] and self.weights.shape == self.lc.shape

    
    def set_knots(self, knotsU=None, multiplicitiesU=None, knotsV=None, multiplicitiesV=None):
        """
        Set knots for curve.

        Parameters
        ----------------------------
        knotsU: list
            A list of the unique knot/node values of the curve along lines of constant V. If set to none, lc will remain unchanged
            (Default: None)
        knotsV: list
            A list of the unique knot/node values of the curve along lines of constant U. If set to none, lc will remain unchanged
            (Default: None)
        multiplicitiesU: list of ints
            A list of containing the number of times each knotU is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline.
            If set to none, lc will remain unchanged
            (Default: None)
        multiplicitiesV: list of ints
            A list of containing the number of times each knotV is repeated. Note that the number of unique knot values * sum(multiplicities) must equal
            the number of control points + degree + 1 and the first and last multiplicity must be equal to degree + 1 to defined a clamped bspline.
            If set to none, lc will remain unchanged
            (Default: None)
        """
        self.knotsU = list(knotsU) if np.any(knotsU) else self.knotsU
        self.knotsV = list(knotsV) if np.any(knotsV) else self.knotsV
        self.multiplicitiesU = list(multiplicitiesU) or self.multiplicitiesU
        self.multiplicitiesV = list(multiplicitiesV) or self.multiplicitiesV

        assert len(self.knotsU) == len(self.multiplicitiesU)
        assert len(self.knotsV) == len(self.multiplicitiesV)

        self.U = list(chain.from_iterable([[knotsU[i]] * multiplicitiesU[i] for i in range(len(knotsU))]))
        self.V = list(chain.from_iterable([[knotsV[i]] * multiplicitiesV[i] for i in range(len(knotsV))]))

    
    def set_degree(self, degreeU=None, degreeV=None):
        """
        Set degree for curve.

        Parameters
        ----------------------------
        degreeU: int
            The degree of the bspline curve in the U direction
        degreeV: int
            The degree of the bspline curve in the V direction
        """
        self.degreeU = degreeU if degreeU != None else self.degreeU
        self.degreeV = degreeV if degreeV != None else self.degreeV


    def set_uniform_lc(self, lc):
        """
        Sets lc for each control point to the value passed in. Lc defines the element size near each control point when the 
        surface is used to defined a mesh using a NURBsGeometry Object. NOTE: Current versions of gmsh don't support using different
        element sizes for different control points - as a result the lc of the first element is the only one that will be used.

        Parameters
        ----------------------------
        lc: float
            Element size
        """
        self.lc = np.ones(self.ctrl_points.shape[0:2]) * lc


    def is_periodic(self):
        """
        Returns whether or not the surface is periodic u and v. The surface is either periodic in u when 
        the first and last control points are equal for all values of v and vice versa.

        Returns
        ----------------------------
        is_u_periodic: bool
            True if the surface is periodic along u, False otherwise
        is_v_periodic: bool
            True if the surface is periodic along v, False otherwise
        """
        is_u_periodic = True
        is_v_periodic = True
        points = self.ctrl_points
        nV = self.nV
        nU = self.nU
        
        for i in range(nV):
            if not np.all(points[i][0] == points[i][-1]):
                is_u_periodic = False
                break

        for i in range(nU):
            if not np.all(points[0][i] == points[-1][i]):
                is_v_periodic = False
                break
        
        return [is_u_periodic, is_v_periodic]


    def get_periodic_multiplicities(self):
        """
        If the surface is periodic in either u or v, then this function returns the periodic 
        multiplicities for either - aka it removes 1 from both the first and last elements in the multiplicities list. Otherwise it just returns
        the multiplicities of the curve.

        Returns
        ----------------------------
        multiplicitiesU: list of int
            If the curve is periodic in u returns the periodic multiplicities of the u knots otherwise returns the unaltered multiplicities
        multiplicitiesV: list of int
            If the curve is periodic in v returns the periodic multiplicities of the v knots otherwise returns the unaltered multiplicities
        """

        is_u_periodic, is_v_periodic = self.is_periodic()

        multiplicitiesU = self.multiplicitiesU[:]
        if is_u_periodic:
            multiplicitiesU[0] -= 1
            multiplicitiesU[-1] -= 1

        multiplicitiesV = self.multiplicitiesV[:]
        if is_v_periodic:
            multiplicitiesV[0] -= 1
            multiplicitiesV[-1] -= 1

        return multiplicitiesU, multiplicitiesV


    def calculate_point(self, u, v): 
        """
        Returns the coordinate of the point at the given value of u and v.

        Parameters
        ----------------------------
        u: float
            Value of the parameter u for which the point will be calculated.
        v: float
            Value of the parameter v for which the point will be calculated.

        Returns
        ----------------------------
        coord: 1d np.array
            The coordinates of the point. If the u value given is less than the first knot or greater than the last
            the coordinate 0 is returned.
        """      
        i_u = find_span(self.nU, self.degreeU, u, self.U)
        i_v = find_span(self.nV, self.degreeV, v, self.V)
        
        N_u = nurb_basis(i_u, self.degreeU, u, self.U)
        N_v = nurb_basis(i_v, self.degreeV, v, self.V)

        temp = np.zeros((self.degreeV+1, self.dim+1))
        for j in range(self.degreeV + 1):
            indU, no_nonzero_basis = find_inds(i_u, self.degreeU)
            indV, _ = find_inds(i_v, self.degreeV)
            for k in range(no_nonzero_basis):
                common = self.weights[indV + j][indU + k] * N_u[k]
                temp[j][-1] += common
                temp[j][0:-1] += self.ctrl_points[indV + j][indU + k] * common
                
        surface = np.zeros(self.dim+1)
        for l in range(self.degreeV + 1):
            surface += N_v[l] * temp[l]

        return surface[:-1]/surface[-1]


    def derivative_wrt_ctrl_point(self, i_deriv, j_deriv, u, v):
        """
        Calculated the derivative of the surface with respect to a control point

        Parameters
        ----------------------------
        i_deriv: int
            The first index of the control point for which you want to calculate the derivative
        j_deriv: int
            The second index of the control point for which you want to calculate the derivative
        u: float
            Value of the parameter u for which at which the derivative will be calculated.
        v: float
            Value of the parameter v for which at which the derivative will be calculated.

        Returns
        ----------------------------
        list [float, 1d numpy array]:
            A list containing the derivative of the curve with repect to the (i_deriv, j_deriv) th control point in the first element and the derivative of the 
            curve with respect to the (i_deriv, j_deriv) th weight in the second element. The derivative of the curve at a point in general is a vector describing
            the direction of movement at that point. A float is returned for the derivative of wrt to the control point where the derivative is equal to the float
            multiplied by the 1 vector.
        """
        i_u = find_span(self.nU, self.degreeU, u, self.U)
        i_v = find_span(self.nV, self.degreeV, v, self.V)

        if i_v < i_deriv or i_v > i_deriv + self.degreeV or i_u < j_deriv or i_u > j_deriv + self.degreeU:
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

        deriv_wrt_point = (N_v[i_deriv - i_v + self.degreeV] * N_u[j_deriv - i_u + self.degreeU] * self.weights[i_deriv][j_deriv]) / denom
        
        deriv_wrt_weight = (N_v[i_deriv - i_v + self.degreeV] * N_u[j_deriv - i_u + self.degreeU] * self.ctrl_points[i_deriv][j_deriv]) / denom - \
                            (numerator_sum * N_v[i_deriv - i_v + self.degreeV] * N_u[j_deriv - i_u + self.degreeU])/denom**2
        
        return [deriv_wrt_point, deriv_wrt_weight]

    
    def derivative_wrt_uv(self, u, v, k):
        """
        Calculates the derivative of the surface with respect to the parameters u and v

        Parameters
        ----------------------------
        u: float
            Value of the parameter u for which at which the derivative will be calculated.
        v: float
            Value of the parameter v for which at which the derivative will be calculated.
        k: int
            The order of the largest derivative you want to calculate

        Returns
        ----------------------------
        ders: np.array
            A numpy array (k+1 x k+1) matrix where each element is contains the derivative of the point with
            respect to the u i times and v j times. Note that the derivative will be in the form of a vector
            and therefore the array will have shape (k+1 x k+1 x self.dim)
        """
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

    
    def get_unit_normal(self, u, v, flip_norm=None):
        """
        Calculates the unit normal to the surface at a given point

        Parameters
        ----------------------------
        u: float
            Value of the parameter u at which the normal will be found
        v: float
            Value of the parameter v at which the normal will be found
        flip_norm: bool
            If true will return the negative of the calculated normal. If None is passed in, the value of flip 
            in self.flip_norm will be used (default: None)

        Returns
        ----------------------------
        unit_norm: np.array
            The unit normal
        """
        flip_norm = flip_norm if flip_norm != None else self.flip_norm

        ders = self.derivative_wrt_uv(u, v, 1)
        norm = np.cross(ders[1][0], ders[0][1])
        unit_norm = norm / np.linalg.norm(norm)

        if flip_norm:
            unit_norm *= -1

        return unit_norm


    def get_unit_normal_derivatives(self, ders):
        """
        Helper function for calculating the curvature. Calculates the derivative of the unit 
        normal at a point on the surface wrt to u and v

        Parameters
        ----------------------------
        ders: float
            ders: np.array
            A numpy array kxk matrix where each element is contains the derivative of the point with
            respect to the u i times and v j times. Note that the derivative will be in the form of a vector.
            This matrix must be at least 2x2 and can be generated using the get_derivatives_wrt_uv function.

        Returns
        ----------------------------
        dn_du: np.array
            The derivative of the unit normal wrt to u
        dn_dv: np.array
            The derivative of the unit normal wrt to v
        """
        dS_du = ders[1][0]
        dS_dv = ders[0][1]
        d2S_du2 = ders[2][0]
        d2S_dv2 = ders[0][2]
        d2S_dudv = ders[1][1]

        nh = np.cross(dS_du, dS_dv)
        nh_mag = np.linalg.norm(nh)
        n = nh/nh_mag

        dnh_du = np.cross(d2S_du2, dS_dv) + np.cross(dS_du, d2S_dudv)
        dnh_dv = np.cross(d2S_dudv, dS_dv) + np.cross(dS_du, d2S_dv2)

        dnh_mag_du = np.dot(n, dnh_du)/nh_mag
        dnh_mag_dv = np.dot(n, dnh_dv)/nh_mag

        dn_du = (dnh_du * nh_mag - nh * dnh_mag_du) / nh_mag**2
        dn_dv = (dnh_dv * nh_mag - nh * dnh_mag_dv) / nh_mag**2

        return [dn_du, dn_dv]


    def get_curvature(self, u, v, flip_norm=False):
        """
        Calculates the curvature of the surface at a point on the surface. The curvature is defined as the divegence of the 
        unit normal along the surface

        Parameters
        ----------------------------
        u: float
            Value of the parameter u for which at which the curvature will be calculated.
        v: float
            Value of the parameter v for which at which the derivative will be calculated.
        flip: False
            Multiplies the result by -1. Equivalent to considering the unit normal in the opposite direction to the one calculated.
            If None is passed in, the value of flip in self.flip_norm will be used (default: None)

        Returns
        ----------------------------
        curvature: float
            The calculated curvature
        """
        ders = self.derivative_wrt_uv(u, v, 2)
        dn_du, dn_dv = self.get_unit_normal_derivatives(ders)

        dS_du = ders[1][0]
        dS_dv = ders[0][1]

        eu = dS_du / np.linalg.norm(dS_du)
        ev = dS_dv / np.linalg.norm(dS_dv)

        ev_perp = (eu - np.dot(eu, ev)*ev)
        ev_perp /= np.linalg.norm(ev_perp)

        eu_perp = (ev - np.dot(ev, eu)*eu)
        eu_perp /= np.linalg.norm(eu_perp)


        curvature = np.dot(dn_du, np.linalg.norm(dS_dv)*ev_perp) + \
                    np.dot(dn_dv, np.linalg.norm(dS_du)*eu_perp)
        curvature /= np.linalg.norm(np.cross(dS_du, dS_dv))

        flip_norm = flip_norm if flip_norm != None else self.flip_norm
        if flip_norm:
            curvature *= -1
        
        return curvature

    
    def get_displacement(self, typ, u, v, i_deriv, j_deriv=0, flip_norm=None, tie=True):
        """
        Calculates the displacement/derivative of a point in the direction normal to the surface wrt a specified parameter

        Parameters
        ----------------------------
        typ: string
            Either "control point" or "weight" depending on which parameter the displacement field will be calculated wrt.
        u: float
            Value of the parameter u at the point at which at which the displacement will be calculated.
        v: float
            Value of the parameter v at the point at which at which the displacement will be calculated.
        i_deriv: int
            The first index of the parameter for which you want to calculate the derivative
        i_deriv: int
            The second index of the parameter for which you want to calculate the derivative (default: 0)
        flip_norm: bool
            If true will consider the displacement in the direction of the opposite unit normal. Note that this is equivalent
            to multiplying the result by -1. If None is passed in, the value of flip in self.flip_norm will be used (default: None)
        tie: bool
            if true the displacement for all control points that are equal to the one indexed will be added to find the displacement field if the 
            points were to all of them were to change together. If typ == "weight", the weights of the points must also match in addtion to the 
            control point. (default: True)

        Returns
        ----------------------------
        displacement: np.array or float
            The displacement of the point. If typ == "control point" then a numpy array is returned where each element is the displacement of the point
            with respect to the kth coordinate. If typ == "weight", the displacement is a float.
        """
        u, v = self.validate_params(u, v)
        
        unit_norm = self.get_unit_normal(u, v, flip_norm=flip_norm)
        if typ == "control point":

            if tie:
                der = 0
                for i, j in self.get_matching_points(i_deriv, j_deriv):
                    der += self.derivative_wrt_ctrl_point(i, j, u, v)[0]
            else:
                 der = self.derivative_wrt_ctrl_point(i_deriv, j_deriv, u, v)[0]

            return unit_norm * der

        elif typ == "weight":

            if tie:
                der = np.zeros(self.dim)
                for i, j in self.get_matching_points(i_deriv, j_deriv):
                    der += self.derivative_wrt_ctrl_point(i, j, u, v)[1]
            else:
                der = self.derivative_wrt_ctrl_point(i_deriv, j_deriv, u, v)[1]

            return np.dot(der, unit_norm)
        
        elif typ == "knot":
            raise NameError("Not implemented")
        else:
            raise NameError(f"No param called {typ}")

    
    def create_ctrl_point_dict(self):
        """
        Creates a dictionary mapping each control point coordinate to its indicies. The dictionary is stored in self.ctrl_points_dict. Keys for the
        dictionary are the coordiates as a tuple and the values are a list of indicies
        """
        for i, row in enumerate(self.ctrl_points):
            for j, point in enumerate(row):
                if tuple(point) in self.ctrl_points_dict:
                    self.ctrl_points_dict[tuple(point)].append((i, j))
                else:
                    self.ctrl_points_dict[tuple(point)] = [(i, j)]

    
    def get_matching_points(self, i, j):
        """
        Returns a list of the indicies of all the control points with the same coordinates as those 
        at the passed in indicies.

        Parameters
        -----------------------------
        i: int
            The first index of the control point for which you want matching indicies
        j: int
            The second index of the control point for which you want matching indicies

        Returns
        ------------------------------
        indicies: list of tuples
            A list of (i, j) tuples which correspond to the indicies of control points with matching coordinates
        """
        if self.ctrl_points_dict == {}:
            self.create_ctrl_point_dict()

        point = self.ctrl_points[i][j]
        if tuple(point) in self.ctrl_points_dict:
             return self.ctrl_points_dict[tuple(point)]
        

    def get_matching_points_and_weights(self, i, j):
        """
        Returns a list of the indicies of all the control points with the same coordinates and weights
        as those at the passed in indicies.

        Parameters
        -----------------------------
        i: int
            The first index of the control point for which you want matching indicies
        j: int
            The second index of the control point for which you want matching indicies

        Returns
        ------------------------------
        indicies: list of tuples
            A list of (i, j) tuples which correspond to the indicies of control points with matching coordinates
            and weights
        """
        if self.ctrl_points_dict == {}:
            self.create_ctrl_point_dict()

        matches = []
        point = self.ctrl_points[i][j]
        if tuple(point) in self.ctrl_points_dict:
             for k, l in self.ctrl_points_dict[tuple(point)]:
                  if self.weights[i][j] == self.weights[k][l]:
                       matches.append((k, l))

        return matches
            

    def calculate_num_and_den_derivatives(self, u, v, k):
        """
        Helper function for get_derivative_wrt_uv. Calculates the derivatives of the numerator
        and denomintor of the expression for the surface.

        Parameters
        -----------------------------
        u: float
            Value of the parameter u for which at which the derivative will be calculated.
        v: float
            Value of the parameter v for which at which the derivative will be calculated.
        k: int
            The order of the largest derivative you want to calculate

        Returns
        ------------------------------
        A_ders: np.array
            A numpy array (k+1 x k+1) matrix where each element is contains the derivative of the numerator with
            respect to the u i times and v j times. Note that the derivative will be in the form of a vector
            and therefore the array will have shape (k+1 x k+1 x self.dim)
        w_ders: np.array
            A numpy array (k+1 x k+1) matrix where each element is contains the derivative of the denominator with
            respect to the u i times and v j times. Note that the derivative will be in the form of a float
            and therefore the array will have shape (k+1 x k+1)
        """
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
                    common = N_ders_u[l][r] * self.weights[i_v-self.degreeV+s][i_u-self.degreeU+r]
                    temp_w[s] += common
                    temp_A[s] += common * self.ctrl_points[i_v-self.degreeV+s][i_u-self.degreeU+r]

            for m in range(min(k-l, min(k, self.degreeV))+1):
                A_ders[l][m] = [0.0] * self.dim
                w_ders[l][m] = 0.0
                for s in range(self.degreeV+1):
                    A_ders[l][m] += N_ders_v[m][s]*temp_A[s]
                    w_ders[l][m] += N_ders_v[m][s]*temp_w[s]

        return A_ders, w_ders


    def validate_params(self, u, v):
        if np.isclose(u, self.knotsU[-1]):
            u = self.knotsU[-1]
        elif np.isclose(u, self.knotsV[0]):
            u = self.knotsU[0]

        if np.isclose(v, self.knotsV[-1]):
            v = self.knotsV[-1]
        elif np.isclose(v, self.knotsV[0]):
            v = self.knotsV[0]

        return [u, v]



def U_vec_to_knots(U):
    """
    Given a knot vector results a list of the unique knot values and a list of the
    multiplicities of each knot.

    Parameters
    -------------------------
    U: list or 1d numpy array
        The knot vector

    Returns
    ------------------------
    knots: list
        The unique knot values in the vector
    multiplicities: list
        The multiplicities (how many times each knot is repeated) of each of the knots in knots
    """
    knots = [U[0]]
    multiplicities = [1]
    for u in U[1:]:
        if knots[-1] != u:
            knots.append(u)
            multiplicities.append(1)
        else:
            multiplicities[-1] += 1

    return knots, multiplicities


def find_inds(i, degree):
    """
    Given a knot span i and a degree returns the index of the 
    first control point with a nonzero basis function in the ith knot span and 
    the number of non-zero basis functions

    Parameters
    -------------------------
    i: int
        The knot span
    degree: int
        The degree of the knot space

    Returns
    ------------------------
    ind: int
        The index of the first control point with a non-zero basis function
    no_nonzero_basis: int
        The number of non-zero basis functions
    """
	
    if i - degree < 0:
        ind = 0
        no_nonzero_basis = i + 1
    else:
        ind = i - degree
        no_nonzero_basis = degree + 1

    return ind, no_nonzero_basis


def find_span(n, p, u, U):
    """
    Given a knot vector, returns the knot span in which a given value of u lies

    Parameters
    -------------------------
    n: int
        The number of control points in the curve
    p: int
        The degree of the knot span
    u: float
        The values of u for which you want to find the knot span
    U: list of float
        The knot vector

    Returns
    ------------------------
    span: int
        The knot span that u lies in
    """
    if u < U[0] or u > U[-1]:
        return -1

    if u == U[n+1]:
        return n-1

    low = p
    high = n+1
    mid = int(np.floor((low + high)/2))
    while(u < U[mid] or u >= U[mid+1]):
        if(u < U[mid]):
            high = mid
        elif(u >= U[mid]):
            low = mid

        mid = int(np.floor((low + high)/2))

    return mid


def nurb_basis(i, p, u, U):
    """
    Given a knot span, returns a list of the values of the non zero basis functions at a given 
    value of U

    Parameters
    -------------------------
    i: int
        The knot span
    p: int
        The degree of the knot span
    u: float
        The values of u at which you want to evaluate the basis functions
    U: list of float
        The knot vector

    Returns
    ------------------------
    N: list of floats
        A list containing the values of the nonzero basis functions at the u
    """
    if i < 0 or i >= len(U):
        return [0] * (p+1)

    left = [0] * (p+1)
    right = [0] * (p+1)
    N = [0] * (p+1)

    N[0] = 1
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u

        saved = 0.0
        for r in range(j):
            temp = N[r]/(right[r+1] + left[j-r])
            N[r] = saved + right[r+1]*temp
            saved = left[j-r]*temp
        
        N[j] = saved

    return N


def nurb_basis_derivatives(i, u, p, U, k):
    """
    Given a knot span, returns a list of the values of the non zero kth derivatives of the basis functions at a given 
    value of u

    Parameters
    -------------------------
    i: int
        The knot span
    p: int
        The degree of the knot span
    u: float
        The values of u at which you want to evaluate the basis functions derivatives
    U: list of float
        The knot vector
    k: int
        The order of the largest derivative you want to calculate

    Returns
    ------------------------
    ders: np.array (k+1 x p+1)
        A numpy array where the ith row contains the ith derivatives of the non-zero basis functions at the given
        value of u.
    """
    ndu = np.zeros((p+1, p+1))
    left = [0] * (p+1)
    right = [0] * (p+1)

    ndu[0][0] = 1.0

    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u

        saved = 0.0
        for r in range(j):
            ndu[j][r] = right[r+1] + left[j-r]
            temp = ndu[r][j-1]/ndu[j][r]

            ndu[r][j] = saved + right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j][j] = saved

    ders = np.zeros((k+1, p+1))
    for j in range(p+1):
        ders[0][j] = ndu[j][p]

    a = np.zeros((2, p+1))
    for r in range(p+1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0
        for l in range(1, k+1):
            d = 0.0
            rl = r-l
            pl = p-l
            if r >= l:
                a[s2][0] = a[s1][0]/ndu[pl+1][rl]
                d = a[s2][0] * ndu[rl][pl]
            
            j1 = 1 if rl >= -1 else -rl
            j2 = l-1 if r-1 <= pl else p-r

            for j in range(j1, j2+1):
                a[s2][j] = (a[s1][j] - a[s1][j-1])/ndu[pl+1][rl+j]
                d += a[s2][j] * ndu[rl+j][pl]

            if r <= pl:
                a[s2][l] = -a[s1][l-1]/ndu[pl+1][r]
                d += a[s2][l] * ndu[r][pl]

            ders[l][r] = d
            j = s1
            s1 = s2
            s2 = j

    r = p
    for l in range(1, k+1):
        for j in range(p+1):
            ders[l][j] *= r
        r *= p-l

    return ders

        

        
