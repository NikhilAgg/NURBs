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
from utils.utils import eigs_to_pdf_1d_robin
from dolfinx.io import XDMFFile

def main():
    #Define Parameters
    N = 100
    k_inlet=1
    k_outlet=1

    #Create Mesh and function space
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = fem.FunctionSpace(msh, ("CG", 1))

    # Boundary Conditions
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1))]
    
    facet_indices, facet_markers = [], []
    fdim = msh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    # Define basis and bilinear form
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    c_squared = fem.Function(V)
    c_squared.interpolate(lambda x: x[0]*0 + 1)
    a = fem.form(
        k_inlet*ufl.inner(u, c_squared*v)*ds(1)
        +
        k_outlet*ufl.inner(u, c_squared*v)*ds(2)
        -
        ufl.inner(ufl.inner(ufl.grad(u), ufl.grad(c_squared)), v)*ufl.dx
        -
        ufl.inner(ufl.grad(u), ufl.grad(c_squared * v))*ufl.dx
    )
    t = fem.form(ufl.inner(u, v)*ufl.dx)

    #Assemble matrix
    A = fem.petsc.assemble_matrix(a)
    A.assemble()

    C = fem.petsc.assemble_matrix(t)
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
    eigs_to_pdf_1d_robin(geometry[:, 0], eig_pairs, ylabel="p", eig_name="$\omega$", filename=path/"output/1D_eigs.pdf")
    print(eig_pairs[2][0])

    # msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    # with XDMFFile(msh.comm, "facet_tags.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(msh)
    #     xdmf.write_meshtags(facet_tag)

if __name__ == "__main__":
    main()