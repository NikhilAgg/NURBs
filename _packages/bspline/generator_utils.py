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
		return np.zeros(p+1)

	left = np.zeros(p+1)
	right = np.zeros(p+1)
	N = np.zeros(p+1)
	
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