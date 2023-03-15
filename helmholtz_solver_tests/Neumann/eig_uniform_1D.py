from dolfinx import fem, mesh, plot
from mpi4py import MPI
import ufl
from petsc4py.PETSc import ScalarType
from slepc4py import SLEPc
import numpy as np
from utils.utils import eigs_to_pdf_1d

def main():
    
    #Define Parameters
    N = 100

    #Create Mesh and function space
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = fem.FunctionSpace(msh, ("CG", 1))

    # Define basis and bilinear form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    c = fem.Constant(V, ScalarType(1))
    a = fem.form(-c**2 * ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx)
    t = fem.form(ufl.inner(u, v)*ufl.dx)

    #Assemble matrix
    A = fem.petsc.assemble_matrix(a)
    A.assemble()

    C = fem.petsc.assemble_matrix(t)
    C.assemble()
    C = -C
    # print(A.getValues(range(N+1), range(N+1)))

    E = SLEPc.EPS()
    E.create()
    E.setOperators(A, C)
    st = E.getST()
    st.setType("sinvert")
    # E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    target = (1*2*np.pi)**2
    E.setTarget(target)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE) #Important target magnitude of the target.
    # E.setFromOptions()
    E.solve()

    #Get Solutions
    nconv = E.getConverged()
    if nconv == 0:
        print("No Solutions Found")
        return
    
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    
    eig_pairs = []
    for i in range(nconv):
        eig = np.sqrt(E.getEigenpair(i, vr, vi))/(2*np.pi)
        eig_pairs.append((eig, vr[:]))

    #- Export and import mesh
    topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    eigs_to_pdf_1d(geometry[:, 0], eig_pairs, ylabel="p")
    

if __name__ == "__main__":
    main()