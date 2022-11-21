from argparse import ArgumentError
import sys
from pathlib import Path
path = Path(__file__).parent.resolve()
sys.path.append(str(path.parents[1]))
from dolfinx import fem, mesh, plot
from dolfinx.io import gmshio
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
from utils.utils import eigs_to_pdf_3d
from dolfinx.io import XDMFFile
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from rijke import create_cylinder


#Define Parameters
N = 20
n = 0.1614
X_h = 0.25
L_h = 0.025
X_w = 0.2
L_w = 0.025
gamma = 1.4
tau = 0.0015
rho_bar = 1.22
p_bar = 1e5
Ru = -0.975+0.05j
Rd = -0.975*0.05j
rhu = 1.4
rhd = 0.9754

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
f4.setRationalNumerator([-1*tau, 0])

f5 = SLEPc.FN()
f5.create()
f5.setType("exp")

f6 = SLEPc.FN()
f6.create()
f6.setType("combine")
f6.setCombineChildren(3, f4, f5)

def create_func(func_space, expr):
    f = fem.Function(func_space)
    if callable(expr) or type(expr) == fem.Expression:
        f.interpolate(expr)
    elif np.isscalar(expr):
        f.interpolate(lambda x: x[0]+expr)
    else:
        raise ArgumentError("Expression must be a constant or a function")

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

def get_lin_eig_solver(A, B, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert"):
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

def get_nonlin_eig_solver(mat_list, func_list, target=(1*2*np.pi)**2, target_space=[], dimensions=1, tol=0.00001):
    if target_space == []:
        target_space = [0, 10*target, -10*target, 10*target]

    N = SLEPc.NEP()
    N.create()
    N.setSplitOperator(mat_list, func_list)
    N.setDimensions(nev=dimensions)
    N.setTolerances(tol=tol)
    N.setTarget(target=target)
    N.setWhichEigenpairs(SLEPc.NEP.Which.TARGET_MAGNITUDE)
    N.setFromOptions()
    N.setType("nleigs")
    rg = N.getRG()
    rg.setType("interval")
    rg.setIntervalEndpoints(*target_space)
    N.solve()
    return N

def get_eigenvalues(eig_solver, A, return_vecs=False):
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    eig_vals = set()
    eigs = []
    
    nconv = eig_solver.getConverged()
    if nconv == 0:
        print("No eigenvalues found")

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

def get_lin_eigenvalues(A, B, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert", return_vecs=False):
    E = get_lin_eig_solver(A, B, target=target, dimensions=dimensions, st_type=st_type)
    eigs = get_eigenvalues(E, A, return_vecs=return_vecs)
    return eigs

def get_poly_eigenvalues(mat_list, target=(1*2*np.pi)**2, dimensions=1, st_type="sinvert", return_vecs=False):
    P = get_poly_eig_solver(mat_list, target=target, dimensions=dimensions, st_type=st_type)
    eigs = get_eigenvalues(P, mat_list[0], return_vecs=return_vecs)
    return eigs

def get_nonlin_eigenvalues(mat_list, func_list, target=(1*2*np.pi)**2, target_space=[], dimensions=1, tol=0.00001, return_vecs=False):
    N = get_nonlin_eig_solver(mat_list, func_list, target=target, target_space=target_space, dimensions=dimensions, tol=tol)
    eigs = get_eigenvalues(N, mat_list[0], return_vecs=return_vecs)
    return eigs

def poly_nonlin_eig_solve(poly_mat_list, D, func, tol=1e-30, error=1e30, maxit=10, init_guess=np.pi):
    A = poly_mat_list[0]
    eigs = get_poly_eigenvalues(poly_mat_list, target=init_guess, dimensions=30)
    for i, eig in enumerate(eigs):
        if abs(eig) > 0.1:
            w_0 = eig
            break

    it = 0
    w_prev = w_0

    while error > tol and it < maxit:
        D_k = func(w_prev)*D
        eigs = get_poly_eigenvalues([A-D_k] + poly_mat_list[1:], target=w_prev, dimensions=3, return_vecs=True)
        w_k = 0
        for eig, vec in eigs:
            if abs(eig) > 0.01:
                w_k = eig
                vec_k = vec
                break
        
        if w_k == 0:
            print(f"No eigenvalues found - iteration: {it}")
            return

        error = abs(w_k - w_prev)/abs(w_0)
        w_prev = w_k
        it += 1

    if it >= maxit:
        print(f"Max iteration reached - Solver did not converge")

    return (w_k, vec_k)

def main():
    #Create Mesh and function space
    msh, cell_markers, facet_markers = gmshio.model_to_mesh(create_cylinder(), MPI.COMM_WORLD, 0, 3)
    # msh = mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
    V = fem.FunctionSpace(msh, ("CG", 1))
    V2 = fem.VectorFunctionSpace(msh, ("CG", 1))

    #Initialise functions
    c_val = np.sqrt(gamma * p_bar / rho_bar)
    c_squared = create_func(V, c_val**2)
    c = create_func(V, c_val)
    h = create_func(V, lambda x: np.exp(-np.square((x[0]-X_h))/L_h**2)/np.sqrt(np.pi)/L_h)
    rh = create_func(V, lambda x: rhu + 0.5*(rhd-rhu)*(1 + np.tanh((x[0]-X_h)/L_h)))

    def make_w(x):
        values = np.zeros((msh.geometry.dim, x.shape[1]))
        values[0] = np.exp(-np.square((x[0]-X_w))/L_w**2)/(np.sqrt(np.pi)*L_w)
        return values

    w = create_func(V2, make_w)
    expr = fem.Expression(w/rh, V.element.interpolation_points())
    w_ro = fem.Function(V2)
    w_ro.interpolate(expr)
    # Boundary Conditions
    # boundaries = [(1, lambda x: np.isclose(x[0], 0)),
    #           (2, lambda x: np.isclose(x[0], 1))]
    
    # ds = tag_boundaries(msh, boundaries)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)

    # Form equations and matricies
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = fem.form(
        -
        ufl.inner(ufl.inner(ufl.grad(u), ufl.grad(c_squared)), v)*ufl.dx
        -
        ufl.inner(ufl.grad(u), ufl.grad(c_squared * v))*ufl.dx
    )
    b_form = fem.form(
        -1j*
        (c)*(Rd+1)/(Rd-1)*ufl.inner(u, c_squared*v)*ds(1)
        +1j*
        (c)*(Ru+1)/(Ru-1)*ufl.inner(u, c_squared*v)*ds(2)
    )
    c_form = fem.form(
        ufl.inner(u, v)*ufl.dx
    )
    d_form = fem.form((gamma-1)*n*ufl.inner(h*ufl.inner(ufl.grad(u), w_ro), v)*ufl.dx)

    A, B, C, D = assemble(a_form, b_form, c_form, d_form)
    # Calculate the eigenvalues
    target = np.pi*c_val/3
    # w_k, vec_k = poly_nonlin_eig_solve([A, B, C], D, lambda x: np.exp(-1j*x*tau), init_guess=target)
    # print(D[20:30,20:30])
    
    eig_pairs = get_nonlin_eigenvalues([A, B, C, -D], [f1, f2, f3, f6], dimensions=5, target=target, return_vecs=True)
    # print(eigs[0][0])
    # eig_pairs = [(w_k, vec_k)]
    # w_k = 3.4255 - 0.0019j
    # # w_k, vec_k = eigs[0]
    # test = A + w_k*B + (w_k**2)*C + np.exp(-1j*w_k*tau)*(-D)
    # print(c_val)
    # y = PETSc.Vec().createSeq(N)
    # y.setValues(range(N), -1*test[1:, 0])
    # x = PETSc.Vec().createSeq(N)
    # x.setValues(range(N), [0]*(N))

    # x = PETSc.Vec().createSeq(N+1)
    # x.setValues(range(N+1), vec_k[:])
    # y = PETSc.Vec().createSeq(N+1)
    # test.mult(x, y)
    # [print(f"{i}: {xi:.5f}") for i, xi in enumerate(y[:])]
    # [print(f"{i}: {xi}") for i, xi in enumerate(y.getValues(range(N+1)))]

    # A = PETSc.Mat().create()
    # A.setSizes([N, N])
    # A.setType("aij")
    # A.setUp()
    # A.setValues(range(N), range(N), test[1:, 1:])
    # A.assemble()

    # ksp = PETSc.KSP().create()
    # ksp.setOperators(A)
    # ksp.setType('bcgs')
    # ksp.setConvergenceHistory()
    # ksp.getPC().setType('none')
    # ksp.solve(y, x)

    # e = PETSc.Vec().createSeq(N+1)
    # e.setValues(range(1, N+1), x.getValues(range(N)))
    # e.setValues(0, 1)
    # y = PETSc.Vec().createSeq(N+1)

    # test.mult(e, y)
    # print((y[:]/e[:]))


    # test.mult(e, y)
    # [print(f"{i}: {xi}") for i, xi in enumerate(y.getValues(range(N+1)))]
    # print(test.getValues(range(N+1), range(N+1)))
    # test.mult(x, y)
    # print(np.round(y.getValues(range(N+1)), 5))
    # eig_solver = lin_eig_solve(A+B, C, dimensions=5)
    # eig_solver = poly_eig_solve([A, B, C], target=2.4, dimensions=60)

    # w_0 = E.getEigenvalue(0)

    # P = SLEPc.PEP()
    # P.create()
    # P.setOperators([A-D])

    # Print Values
    # if msh.comm.rank == 0:
    #     for i in 
    #     print(f"\n\ns: {np.sqrt(eig_pairs[i][0])}\n\n")

    # # #- Export and import mesh
    topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    eigs_to_pdf_3d(V, msh, eig_pairs, data_label="p", filename=path/"output/3D_eigs.pdf")

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