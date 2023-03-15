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
from utils.utils import eigs_to_pdf_1d, eigs_to_pdf_3d
import matplotlib.pyplot as plt
import pyvista
import matplotlib as mpl


def main():
    #Define Parameters
    N = 30

    #Create Mesh and function space
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 100, N, N)
    V = fem.FunctionSpace(msh, ("CG", 1))

    #Boundary Condtions
    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1))]
            #   (3, lambda x: np.isclose(x[1], 0)), 
            #   (4, lambda x: np.isclose(x[1], 1))]
    
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
        # -
        # ufl.inner(u, c_squared*v)*ds(4)
        # -
        # ufl.inner(u, c_squared*v)*ds(3)
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
    E.setDimensions(nev=30)
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
        print(f"\n\ns: {[i[0] for i in eig_pairs]}\n\n")

    eigs_to_pdf_3d(V, msh, eig_pairs, data_label="p", filename=path/"output/3D_eigs.pdf")
    
    # #Plot
    # for i in range(nconv):
    #     p = fem.Function(V)
    #     p.vector[:] = eig_pairs[i][1]
    #     plotter = pyvista.Plotter(shape=(1, 3))
    #     plotter.add_axes(line_width=5)

        

    #     #- Export and import mesh
    #     topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    #     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    #     #- Export and import solution
    #     u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    #     u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    #     u_grid.point_data["u"] = p.x.array.real
    #     u_grid.set_active_scalars("u")
    #     u_slices = pyvista.MultiBlock()
    #     # u_slices.append(u_grid.slice(normal=(-1, 1, 1), origin=grid.center))
    #     # u_slices.append(u_grid.slice(normal=(1, 1, 1), origin=grid.center))
    #     u_slices.append(u_grid.slice_along_axis(n=5, axis="x"))
    #     u_slices.append(u_grid.slice_along_axis(n=5, axis="y")) 


    #     ncolors = 256
    #     cmap = plt.cm.get_cmap("PiYG")(range(ncolors))

    #     cmap[:,-1] = np.concatenate([np.linspace(1.0, 0, ncolors//2), np.linspace(0, 1, ncolors//2)])
    #     map_object = mpl.colors.LinearSegmentedColormap.from_list(name='mine',colors=cmap)
    #     cmap=map_object

    #     max_u = np.max(np.abs(p.x.array.real))
    #     outline = grid.outline()
    #     for i in range(3):
    #         plotter.subplot(0, i)
    #         plotter.add_mesh(outline)
    #         plotter.add_mesh(u_slices, show_edges=False, cmap=cmap, opacity=1, clim=[-max_u, max_u])
    #         #- Show the plot
        
            
    #         plotter.camera_position = 'yz'
    #         plotter.camera.azimuth = 60*i + 30
    #         plotter.camera.elevation = 30
    #     plotter.show()

if __name__ == "__main__":
    main()