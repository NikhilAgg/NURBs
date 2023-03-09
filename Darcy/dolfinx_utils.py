from dolfinx import mesh, fem
from slepc4py import SLEPc
import ufl
from dolfinx.io import XDMFFile
import numpy as np
from mpi4py import MPI


def create_func(func_space, expr):
    f = fem.Function(func_space)
    if callable(expr) or type(expr) == fem.Expression:
        f.interpolate(expr)
    elif np.isscalar(expr):
        f.interpolate(lambda x: x[0]+expr)
    else:
        raise ValueError("Expression must be a constant or a function")

    return f
    
def tag_boundaries(msh, boundaries, return_tags=False):
    facet_indices, facet_markers = [], []
    fdim = msh.topology.dim - 1

    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(msh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))

    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = mesh.meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    if return_tags:
        return (ds, facet_tags)
    else:
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

def get_nonlin_eig_solver(mat_list, func_list, target=(1*2*np.pi)**2, target_space=[], dimensions=1, tol=1e-14):
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
            eigs.append((eig, (vr[:], vi[:])))
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

def get_nonlin_eigenvalues(mat_list, func_list, target=(1*2*np.pi)**2, target_space=[], dimensions=1, tol=1e-14, return_vecs=False):
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
                print(f"Current Eigenvalue: {w_k}")
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

def XDMF_reader(name):
    mesh_loader_name = name + ".xdmf"
    tag_loader_name = name + "_tags.xdmf"
    with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")    

    return [mesh, cell_tags, facet_tags]

def gaussian3D(x,x_ref,sigma):
    # https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
    N = 1/(sigma**3*(2*np.pi)**(3/2))
    spatials = np.square(x[0]-x_ref[0]) + np.square(x[1]-x_ref[1]) + np.square(x[2]-x_ref[2])
    exp = np.exp(-spatials/(2*sigma**2))
    return N*exp

def normalize(func):
    integral_form = fem.form(func*ufl.dx)
    integral= MPI.COMM_WORLD.allreduce(fem.assemble_scalar(integral_form), op=MPI.SUM)

    func.x.array[:] /= integral
    func.x.scatter_forward()

    return func

def normalize_magnitude(func, imaginary=False):
    if imaginary:
        func.vector[:] = func.vector[:] / func.x.array[np.argmax(abs(func.x.array))]
    else:
        func.vector[:] = func.vector[:].real / func.x.array[np.argmax(abs(func.x.array.real))].real

    return func