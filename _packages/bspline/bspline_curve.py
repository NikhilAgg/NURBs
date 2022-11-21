import numpy as np

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

        
