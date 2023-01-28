import numpy as np
import math
from itertools import chain

class NURBsCurve:
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

        #Other
        self.ctrl_point_dict = {}

        #TODO FIX U DIFFERENCE AND PERIODIC SPLINES
        #TODO ADD CHECKS
        

    def set_uniform_lc(self, lc):
        self.lc = np.ones(self.n) * lc


    def is_closed(self):
        return np.all(self.ctrl_points[0] == self.ctrl_points[-1])


    def calculate_point(self, u):
        P_w = np.column_stack(((self.ctrl_points.T * self.weights).T, self.weights))

        i = find_span(self.n, self.degree, u, self.U)
        N = nurb_basis(i, self.degree, u, self.U)

        curve = np.zeros(len(P_w[0]))

        ind, no_nonzero_basis = find_inds(i, self.degree)
        for k in range(no_nonzero_basis):
            curve += P_w[ind + k] * N[k]

        return curve[:-1]/curve[-1]


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

    
    def get_unit_normal(self, u, flip=False):
        dC = self.derivative_wrt_u(u, 1)[1]
        dC_hat = dC / np.linalg.norm(dC)

        temp = dC_hat[0]
        dC_hat[0] = -dC_hat[1]
        dC_hat[1] = temp

        if flip:
            dC_hat *= -1

        return dC_hat

    
    def get_curvature(self, u, flip=False):
        ders = self.derivative_wrt_u(u, 2)
        dC = ders[1]
        d2C = ders[2]

        curvature = np.linalg.norm(np.cross(dC, d2C)) / np.linalg.norm(dC)**3

        if flip:
            curvature *= -1

        return curvature


    def get_displacement_field(self, u, i_deriv, param, flip=False):
        
        if param == "control point":
            magnitude = self.derivative_wrt_ctrl_point(self, i_deriv, u)[0]
            der = np.ones(self.dim) * magnitude / np.sqrt(self.dim)
        elif param == "weight":
            der = self.derivative_wrt_ctrl_point(self, i_deriv, u)[1]
        elif param == "knot":
            raise NameError("Not implemented")
        else:
            raise NameError(f"No param called {param}")

        unit_norm = self.get_unit_normal(u, flip=flip)

        return np.dot(der, unit_norm)


    def create_ctrl_point_dict(self):
        for i, point in enumerate(self.ctrl_points):
            if tuple(point) in self.ctrl_points_dict:
                self.ctrl_points_dict.append(i)
            else:
                self.ctrl_points_dict[point] = [i]
        

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



class NURBsSurface:
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

        #Other
        self.ctrl_points_dict = {}
        

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

    
    def get_unit_normal(self, u, v, flip=False):
        ders = self.derivative_wrt_uv(u, v, 1)
        norm = np.cross(ders[1][0], ders[0][1])
        unit_norm = norm / np.linalg.norm(norm)

        if flip:
            unit_norm *= -1

        return unit_norm


    def get_unit_normal_derivatives(self, ders):
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


    def get_curvature(self, u, v, flip=False):
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

        if flip:
            curvature *= -1
        
        return curvature

    
    def get_displacement_field(self, u, v, i_deriv, j_deriv, param, flip=False):
        if param == "control point":
            magnitude = self.derivative_wrt_ctrl_point(self, i_deriv, j_deriv, u, v)[0]
            der = np.ones(self.dim) * magnitude / np.sqrt(self.dim)
        elif param == "weight":
            der = self.derivative_wrt_ctrl_point(self, i_deriv, j_deriv, u, v)[1]
        elif param == "knot":
            raise NameError("Not implemented")
        else:
            raise NameError(f"No param called {param}")

        unit_norm = self.get_unit_normal(u, v, flip=flip)

        return np.dot(der, unit_norm)

    
    def create_ctrl_point_dict(self):
        for i, row in enumerate(self.ctrl_points):
            for j, point in enumerate(row):
                if tuple(point) in self.ctrl_points_dict:
                    self.ctrl_points_dict.append((i, j))
                else:
                    self.ctrl_points_dict[point] = [(i, j)]


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


def find_inds(i, degree):
	if i - degree < 0:
		ind = 0
		no_nonzero_basis = i + 1
	else:
		ind = i - degree
		no_nonzero_basis = degree + 1

	return ind, no_nonzero_basis


def find_span(n, p, u, U):
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

        

        
