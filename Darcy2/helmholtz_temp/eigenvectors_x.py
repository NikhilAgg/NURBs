from dolfinx.fem.assemble import assemble_scalar
import numpy as np
from slepc4py import SLEPc
from dolfinx.fem import Function, FunctionSpace, form
from ufl import dx
from .petsc4py_utils import multiply, vector_matrix_vector, matrix_vector
from mpi4py import MPI
from petsc4py import PETSc

def normalize_eigenvector(mesh, obj, i, degree=1, which='right',mpc=None):
    """ 
    This function normalizes the eigensolution vr
     which is obtained from complex slepc build
     (vi is set to zero in complex build) 

    Args:
        mesh ([dolfinx.cpp.mesh.Mesh]): mesh of the domain
        vr ([petsc4py.PETSc.Vec]): eigensolution
        degree (int, optional): degree of finite elements. Defaults to 1.

    Returns:
        [<class 'dolfinx.fem.function.Function'>]: normalized eigensolution such that \int (p p dx) = 1
    """

    A = obj.getOperators()[0]
    vr, vi = A.createVecs()
    
    if isinstance(obj, SLEPc.EPS):
        eig = obj.getEigenvalue(i)
        omega = np.sqrt(eig)
        if which == 'right':
            obj.getEigenvector(i, vr, vi)
        elif which == 'left':
            obj.getLeftEigenvector(i, vr, vi)
    
    elif isinstance(obj, SLEPc.PEP):
        eig = obj.getEigenpair(i, vr, vi)
        omega = eig

    if mpc:
        vr = matrix_vector(mpc,vr)

    V = FunctionSpace(mesh, ("Lagrange", degree))
    p = Function(V)

    p.vector.setArray(vr.array)
    p.x.scatter_forward()

    meas = np.sqrt(mesh.comm.allreduce(assemble_scalar(form(p*p*dx)), op=MPI.SUM))
    
    temp = vr.array
    temp= temp/meas

    p_normalized = Function(V) # Required for Parallel runs
    p_normalized.name = "P"
    p_normalized.vector.setArray(temp)
    p_normalized.x.scatter_forward()

    if MPI.COMM_WORLD.rank == 0:
        print(f"Eigenvalue-> {omega:.6f}\n")
        print(f"Frequency -> {omega.real/(2*np.pi):.6f} (1/s) | Growth Rate -> {omega.imag:.6f} (rad/s) \n")

    return omega, p_normalized

def normalize_adjoint(omega_dir, p_dir, p_adj, matrices, D=None):
    """
    Normalizes adjoint eigenfunction for shape optimization.

    Args:
        omega_dir ([complex]): direct eigenvalue
        p_dir ([<class 'dolfinx.fem.function.Function'>]): direct eigenfunction
        p_adj ([<class 'dolfinx.fem.function.Function'>]): adjoint eigenfunction
        matrices ([type]): passive_flame object
        D ([type], optional): active flame matrix

    Returns:
        [<class 'dolfinx.fem.function.Function'>]: [description]
    """
    

    B = matrices.B

    p_dir_vec = p_dir.vector
    p_adj_vec = p_adj.vector

    if not B and not D:
        print('not B and not D: return None')
        return p_adj
    elif B and not D:
        # B + 2 \omega C
        dL_domega = (B +
                     matrices.C * (2 * omega_dir))
    elif D and not B:
        # 2 \omega C - D'(\omega)
        dL_domega = (matrices.C * (2 * omega_dir) -
                     D.get_derivative(omega_dir))
    else:
        # B + 2 \omega C - D'(\omega)
        dL_domega = (B +
                     matrices.C * (2 * omega_dir) -
                     D.get_derivative(omega_dir))

    meas = vector_matrix_vector(p_adj_vec, dL_domega, p_dir_vec)

    p_adj_vec = multiply(p_adj_vec, 1 / meas)

    p_adj1 = p_adj
    p_adj1.name = "P_Adj"
    p_adj1.vector.setArray(p_adj_vec.getArray())
    p_adj1.x.scatter_forward()

    return p_adj1

def normalize_unit(P):

    x = P.vector
    comm = x.getComm()
    rank = comm.getRank()
    n = 1 if rank == 0 else 0
    aux = PETSc.Vec().createMPI((n, PETSc.DECIDE), comm=comm)
    if rank == 0: aux[0] = x[0]
    aux.assemble()
    x0 = aux.sum()
    print(x0)
    sign = x0/abs(x0)
    x.scale(1.0/sign)

    P.x.array[:] = x.array[:]

    return P