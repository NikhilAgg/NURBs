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
from dolfinx.io import XDMFFile
import matplotlib.pyplot as plt

#Define Parameters
N = 50
eta = 0.1614
X_h = 0.25
L_h = 0.025
X_w = 0.2
L_w = 0.025
gamma = 1.4
tau = 0.0015
rhobar=1.22
Ru = 0
Rd = 0

def create_func(func_space, func):
    f = fem.Function(func_space)
    f.interpolate(func)
    return f
    
def tag_boundaries(msh, boundaries):
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
    
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

    return ds

def assemble(*args):
    Mats = []
    for form in args:
        Mat = fem.petsc.assemble_matrix(form)
        Mat.assemble()
        Mats.append(Mat)
    return Mats

def lin_eig_solver(A, B, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert"):
    E = SLEPc.EPS()
    E.create()
    st = E.getST()
    st.setType(st_type)
    E.setOperators(A, B)
    E.setTarget(target)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    E.setDimensions(nev=dimensions)
    E.solve()

    return E

def poly_eig_solve(mat_list, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert"):
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

def main():
    #Create Mesh and function space
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, N)
    V = fem.FunctionSpace(msh, ("CG", 1))
    V2 = fem.VectorFunctionSpace(msh, ("CG", 1))

    #Initialise functions
    c_squared = create_func(V, lambda x: x[0]*0 + 1)
    c = create_func(V, lambda x: x[0]*0 + 1)
    h = create_func(V, lambda x: np.exp(-np.square((x[0]-X_h))/L_h**2)/np.sqrt(np.pi)/L_h)

    def make_w_ro(x):
        values = np.zeros((msh.geometry.dim, x.shape[1]))
        values[0] = (1/rhobar)*np.exp(-np.square((x[0]-X_w))/L_w**2)/np.sqrt(np.pi)/L_w
        return values[0]

    w_ro = create_func(V2, make_w_ro)

    # Boundary Conditions
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1))]
    
    # Define basis and bilinear form
    ds = tag_boundaries(msh, boundaries)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = fem.form(
        -
        ufl.inner(ufl.inner(ufl.grad(u), ufl.grad(c_squared)), v)*ufl.dx
        -
        ufl.inner(ufl.grad(u), ufl.grad(c_squared * v))*ufl.dx
    )
    b_form = fem.form(
        -
        (1/c)*(Rd-1)/(Rd+1)*ufl.inner(u, c_squared*v)*ds(1)
        +
        (1/c)*(Ru-1)/(Ru+1)*ufl.inner(u, c_squared*v)*ds(2)
    )
    c_form = fem.form(
        -ufl.inner(u, v)*ufl.dx
    )
    d_form = fem.form(-(gamma-1)*ufl.inner(h*ufl.inner(ufl.grad(u), w_ro), v)*ufl.dx)

    #Assemble matrix
    A, B, C, D = assemble(a_form, b_form, c_form, d_form)


    # eig_solver = lin_eig_solve(A+B, C, dimensions=5)
    eig_solver = poly_eig_solve([A, B, C], target=2.4, dimensions=60)

    # w_0 = E.getEigenvalue(0)

    # P = SLEPc.PEP()
    # P.create()
    # P.setOperators([A-D])

    #Get Solutions
    nconv = eig_solver.getConverged()
    if nconv == 0:
        print("No Solutions Found")
        return
    
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()
    
    eig_pairs = []
    for i in range(nconv):
        eig = eig_solver.getEigenpair(i, vr, vi)
        print(eig)
        # print(eig_solver.computeError(i))
        eig_pairs.append((eig, vr[:]))

    
    # Print Values
    # if msh.comm.rank == 0:
    #     for i in 
    #     print(f"\n\ns: {np.sqrt(eig_pairs[i][0])}\n\n")

    # # #- Export and import mesh
    topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    eigs_to_pdf_1d(geometry[:, 0], eig_pairs, ylabel="p", eig_name="$\omega$", filename=path/"output/1D_eigs.pdf")

    # msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    # with XDMFFile(msh.comm, "facet_tags.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(msh)
    #     xdmf.write_meshtags(facet_tag)

if __name__ == "__main__":
    main()



    # f1 = SLEPc.FN()
    # f1.create()
    # f1.setType("rational")
    # f1.setRationalNumerator([1])

    # f2 = SLEPc.FN()
    # f2.create()
    # f2.setType("rational")
    # f2.setRationalNumerator([1, 0, 0])
    
    # f3 = SLEPc.FN()
    # f3.create()
    # f3.setType("rational")
    # f3.setRationalNumerator([-1j*tau, 0])
    

    # f4 = SLEPc.FN()
    # f4.create()
    # f4.setType("exp")

    # f5 = SLEPc.FN()
    # f5.create()
    # f5.setType("combine")
    # f5.setCombineChildren(3, f3, f4)
    # print(f5.evaluateFunction(np.pi/tau))


    # N = SLEPc.NEP()
    # N.create()
    # N.setSplitOperator([A, B, C], [f1, f5, f3])
    # # N.setTarget(0.001936j + 3.425376)
    # # N.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    # N.setDimensions(nev=10)
    # N.solve()