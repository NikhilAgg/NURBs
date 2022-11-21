import sys
from pathlib import Path
path = Path(__file__).parent.resolve()
sys.path.append(str(path.parents[1]))
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
    c_squared = fem.Function(V)
    c_squared.interpolate(lambda x: x[0]*0 + 1)
    a = fem.form(-1 * (
        ufl.inner(ufl.inner(ufl.grad(u), ufl.grad(c_squared)), v)*ufl.dx
        +
        ufl.inner(ufl.grad(u), ufl.grad(c_squared * v))*ufl.dx
    ))
    t = fem.form(ufl.inner(u, v)*ufl.dx)

    #Boundary Conditions
    bound_func = fem.Function(V)
    # bound_func.interpolate(lambda x: x[0]*0)

    def onbound(x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

    bound_facets = mesh.locate_entities_boundary(msh, msh.topology.dim-1, onbound)
    bound_dof = fem.locate_dofs_topological(V, msh.topology.dim-1, bound_facets)

    bound_cond = fem.dirichletbc(bound_func, bound_dof)

    #Assemble matrix
    A = fem.petsc.assemble_matrix(a, bcs=[bound_cond])
    A.assemble()

    C = fem.petsc.assemble_matrix(t, bcs=[bound_cond])
    C.assemble()
    C = -C

    E = SLEPc.EPS()
    E.create()
    E.setOperators(A, C)
    st = E.getST()
    st.setType("sinvert")
    target = (1*2*np.pi)**2
    E.setTarget(target)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setDimensions(nev=5)
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
    #Print Values
    # if msh.comm.rank == 0:
        # print(f"\n\ns: {np.sqrt(eig_vals)}\n\n")

    #- Export and import mesh
    topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    eigs_to_pdf_1d(geometry[:, 0], eig_pairs, ylabel="p", eig_name="$\omega$", filename=path/"output/1D_eigs.pdf")
    print(eig_pairs[2][1])

if __name__ == "__main__":
    main()