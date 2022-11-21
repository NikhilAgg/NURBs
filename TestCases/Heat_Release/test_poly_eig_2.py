import petsc4py.PETSc as PETSc
from slepc4py import SLEPc
import numpy as np
from dolfinx import mesh, fem

P = SLEPc.PEP()
P.create()
st = P.getST()
st.setType("sinvert")
P.setOperators([A, B, C])
P.setTarget(1.5)
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