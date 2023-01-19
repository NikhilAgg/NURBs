import numpy as np

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

