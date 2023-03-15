from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

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

f1 = SLEPc.FN()
f1.create()
f1.setType("rational")
f1.setRationalNumerator([1])

f2 = SLEPc.FN()
f2.create()
f2.setType("rational")
f2.setRationalNumerator([1, 0])

f3 = SLEPc.FN()
f3.create()
f3.setType("rational")
f3.setRationalNumerator([1, 0, 0])

f4 = SLEPc.FN()
f4.create()
f4.setType("rational")
f4.setRationalNumerator([-1, 0])

f5 = SLEPc.FN()
f5.create()
f5.setType("exp")

f6 = SLEPc.FN()
f6.create()
f6.setType("combine")
f6.setCombineChildren(3, f4, f5)
# print(f5.evaluateFunction(np.pi/tau/2))


N = SLEPc.NEP()
N.create()
N.setSplitOperator([A, B, C, D], [f1, f2, f3, f6])
# N.setProblemType(1)
# N.setTarget(0.001936j + 3.425376)
# N.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
N.setDimensions(nev=1)
N.setTolerances(tol=0.001)
N.setWhichEigenpairs(SLEPc.NEP.Which.TARGET_MAGNITUDE)
N.setTarget(target=50)
# N.setType("rii")
# N.setInitialSpace(C)
# N.errorView()
# N.view()
N.setFromOptions()
N.solve()

#Get Solutions
nconv = N.getConverged()
if nconv == 0:
    print("No Solutions Found")

vr, wr = A.getVecs()
vi, wi = A.getVecs()

eig_pairs = []
print(f"Number of converged solutions: {nconv}\n")
for i in range(nconv):
    # eig = np.sqrt(N.getEigenpair(i, vr, vi))/(2*np.pi)
    eig = N.getEigenpair(i, vr, vi)
    print(f"Eigenvalue: {eig}")
    w_k = eig
    det = (1+w_k)*(w_k**2 + np.exp(-w_k)) - 1
    print(f"Detirminant: {det}")
    print("\n\n")
    eig_pairs.append((eig, vr[:]))


test = A + w_k*B + (w_k**2)*C + np.exp(-w_k)*D
print(test[:, :])