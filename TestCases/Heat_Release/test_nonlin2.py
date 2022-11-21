import petsc4py.PETSc as PETSc
from slepc4py import SLEPc
import numpy as np
from dolfinx import mesh, fem

A = PETSc.Mat().create()
A.setSizes([2, 2])
A.setType("aij")
A.setUp()
A.setValues([0, 1], [0, 1], np.array([[1, 1], [1, 1]]))
A.assemble()

B = PETSc.Mat().create()
B.setSizes([2, 2])
B.setType("aij")
B.setUp()
B.setValues([0, 1], [0, 1], np.array([[1, 0], [0, 0]]))
B.assemble()

C = PETSc.Mat().create()
C.setSizes([2, 2])
C.setType("aij")
C.setUp()
C.setValues([0, 1], [0, 1], np.array([[0, 0], [0, 1]]))
C.assemble()

D = PETSc.Mat().create()
D.setSizes([2, 2])
C.setType("aij")
D.setUp()
D.setValues([0, 1], [0, 1], np.array([[0, 0], [0, 1]]))
D.assemble()
D = -D


def get_poly_eig_solver(mat_list, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert"):
    P = SLEPc.PEP()
    P.create()
    st = P.getST()
    st.setType(st_type)
    P.setOperators(mat_list)
    P.setTarget(target)
    # P.setInterval(0.1, 50)
    P.setProblemType(SLEPc.PEP.ProblemType.GENERAL)
    P.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    P.setDimensions(nev=dimensions)
    P.setConvergenceTest(SLEPc.PEP.Conv.NORM)
    # P.setTolerances(1e-20, 1e4)
    P.solve()

    return P

def get_eigenvalues(eig_solver, return_vecs=False):
    A = eig_solver.getOperators()[0]
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    eig_vals = set()
    eigs = []
    
    nconv = eig_solver.getConverged()
    for i in range(nconv):
        eig = np.round(eig_solver.getEigenpair(i, vr, vi), 5)

        if eig in eig_vals:
            continue

        if return_vecs == True:
            eigs.append((eig, vr[:]))
        else:
            eigs.append(eig)

    if return_vecs == True:
        eigs = sorted(eigs, key=lambda x: abs(x[0]))
    else:
        eigs = sorted(eigs, key=lambda x: abs(x))

    return eigs

def get_poly_eigenvalues(mat_list, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert", return_vecs=False):
    P = get_poly_eig_solver(mat_list, target=target, dimensions=dimensions, st_type="sinvert")
    eigs = get_eigenvalues(P, return_vecs=return_vecs)

    return eigs

eigs = get_poly_eigenvalues([A, B, C], target=0.7548, dimensions=30)
for i, eig in enumerate(eigs):
    if abs(eig) > 0.1:
        ind = i
        break

w_0 = eigs[ind]

tol = 1e-30
error = 1e30
maxit = 10
it = 0
w_prev = w_0
while error > tol and it < maxit:
    D_k = np.exp(-w_prev)*D
    eigs = get_poly_eigenvalues([A-D_k, B, C], target=w_prev, dimensions=3, return_vecs=True)
    w_k = 0
    for eig, vec in eigs:
        if abs(eig):
            w_k = eig
            vec_k = vec
            break
    
    if w_k == 0:
        print(f"No eigenvalues found - iteration: {it}")

    error = abs(w_k - w_prev)/abs(w_0)
    w_prev = w_k
    print(w_k)
    it += 1

# print(test[:,:])
# print("\n\n\n")
# print((1+w_k)*(w_k**2 + np.exp(-w_k)) - 1)

test = A + w_k*B + (w_k**2)*C + np.exp(-w_k)*(-D)
print(test[:, :])