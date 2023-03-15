import petsc4py.PETSc as PETSc
from slepc4py import SLEPc
import numpy as np

tau = 1

A = PETSc.Mat().create()
A.setSizes([2, 2])
A.setType("aij")
A.setUp()

A.setValues([0, 1], [0, 1], np.array([[-1, 1], [1, -1]]))
A.assemble()

B = PETSc.Mat().create()
B.setSizes([2, 2])
B.setType("aij")
B.setUp()

B.setValues([0, 1], [0, 1], np.array([[1, 0], [0, -1]]))
B.assemble()


C = PETSc.Mat().create()
C.setSizes([2, 2])
C.setType("aij")
C.setUp()

C.setValues([0, 1], [0, 1], np.array([[-0.3333333, -0.1666667], [-0.16666667, -0.3333333333]]))
C.assemble()
C = PETSc.Vec().create()
C.setSizes(2)
C.setUp()
C.setValues([1, 2], [1, 1])
C.assemble()

# test = np.array(A.getValues(range(2), range(2)))
# print(np.linalg.det(test))

P = SLEPc.PEP()
P.create()
st = P.getST()
st.setType("sinvert")
P.setOperators([A, B, C])
P.setTarget(1j)
P.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
P.setDimensions(nev=1)
P.solve()

#Get Solutions
nconv = P.getConverged()
if nconv == 0:
    print("No Solutions Found")

vr, wr = A.getVecs()
vi, wi = A.getVecs()

eig_pairs = []
for i in range(nconv):
    # eig = np.sqrt(N.getEigenpair(i, vr, vi))/(2*np.pi)
    eig = P.getEigenpair(i, vr, vi)
    print(eig)
    eig_pairs.append((eig, vr[:]))
