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
from utils.utils import eigs_to_pdf_2d_robin
import matplotlib.pyplot as plt
import pyvista
from dolfinx.io import XDMFFile


def main():
    #Define Parameters
    N = 100

    #Create Mesh and function space
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.FunctionSpace(msh, ("CG", 1))

    #Boundary Conditions
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

    # Define basis and bilinear form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    c_squared = fem.Function(V)
    c_squared.interpolate(lambda x: x[0]*0 + 1)
    a = fem.form(
        ufl.inner(u, c_squared*v)*ds(2)
        -
        ufl.inner(u, c_squared*v)*ds(1)
        -
        ufl.inner(ufl.inner(ufl.grad(u), ufl.grad(c_squared)), v)*ufl.dx
        -
        ufl.inner(ufl.grad(u), ufl.grad(c_squared * v))*ufl.dx
    )
    t = fem.form(ufl.inner(u, v)*ufl.dx)

    # #Boundary Conditions
    # bound_func = fem.Function(V)
    # bound_func.interpolate(lambda x: x[0]*0)

    # def onbound(x):
    #     return np.logical_or(np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)), 
    #                                 np.logical_or(np.isclose(x[1], 1), np.isclose(x[1], 0)))

    # bound_facets = mesh.locate_entities_boundary(msh, msh.topology.dim-1, onbound)
    # bound_dof = fem.locate_dofs_topological(V, msh.topology.dim-1, bound_facets)

    # bound_cond = fem.dirichletbc(bound_func, bound_dof)

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
    E.setDimensions(nev=10)
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
    if msh.comm.rank == 0:
        print(f"\n\ns: {np.sqrt([i[0] for i in eig_pairs])}\n\n")

    
    eigs_to_pdf_2d_robin(V, msh, eig_pairs, data_label="p", filename=path/"output/2D_eigs.pdf")
    # Plot
    # print(nconv)
    # for i in range(5, 6):
    #     p = fem.Function(V)
    #     p.vector[:] = eig_pairs[i][1]
    #     plotter = pyvista.Plotter()
    #     plotter.add_axes(line_width=5)

    #     #- Export and import mesh
    #     topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    #     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    #     plotter.add_mesh(grid, show_edges=True)

    #     #- Export and import solution
    #     u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    #     u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    #     u_grid.point_data["u"] = p.x.array.real
    #     u_grid.set_active_scalars("u")
    #     plotter.add_mesh(u_grid, show_edges=True)

        # #- Show the plot
        # plotter.view_xy()
        # plotter.show()
        # x = np.linspace(0, 1, 100)
        # i=1
        # p = u_grid.sample_over_line((0, 0.5, 0), (1, 0.5, 0), resolution=99).point_data["u"]
        # p_act = (np.sin(np.pi*i*x) + i*np.pi*np.cos(i*np.pi*x))
        # if np.max(np.abs(p_act)) != 0:
        #     p_act = p_act*(np.max(np.abs(p))/np.max(np.abs(p_act))) 
        # plt.plot(x, p_act)
        # plt.plot(x, p)
        # plt.show()
        
    msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    with XDMFFile(msh.comm, "facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(facet_tag)

if __name__ == "__main__":
    main()