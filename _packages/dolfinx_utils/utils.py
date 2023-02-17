import matplotlib.backends.backend_pdf as pltpdf
import matplotlib.pyplot as plt
import matplotlib.colors as mplclr
import numpy as np
import pyvista
from dolfinx import fem, plot

def eigs_to_pdf_1d(x, eig_pairs, xlabel="x", ylabel="y", eig_name="Eigenvalue", filename="output.pdf"):
    eig_pairs.sort()
    pdf = pltpdf.PdfPages(filename)
    for i in range(len(eig_pairs)):
        if np.isclose(abs(eig_pairs[i][0]), 0):
            continue
        plt.plot(x, np.real(eig_pairs[i][1]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{eig_name}: {eig_pairs[i][0]:.2f}")
        pdf.savefig()
        plt.clf()
    pdf.close()


def eigs_to_pdf_2d(V, mesh, eig_pairs, data_label="u", eig_name="Eigenvalue", filename="output.pdf"):
    eig_pairs.sort()

    pdf = pltpdf.PdfPages(filename)
    plotter = pyvista.Plotter(off_screen=True)
    plotter.store_image = True
    plotter.ren_win.SetSize([2500, 1500])
    plt.axis("off")

    topology, cell_types, geometry = plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid, show_edges=True)

    #- Export and import solution
    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    plotter.add_axes(line_width=5)
    cmap = plt.cm.get_cmap("PiYG")

    p = fem.Function(V)
    for i in range(len(eig_pairs)):
        if np.isclose(eig_pairs[i][0].real, 0):
            continue
        p.vector[:] = np.real(eig_pairs[i][1])
        u_grid.point_data[data_label] = p.x.array.real
        u_grid.set_active_scalars(data_label)

        max_u = np.max(np.abs(p.vector))
        actor = plotter.add_mesh(u_grid, show_edges=True, clim=[-max_u, max_u], cmap=cmap)
        plotter.view_xy()
        plt.imshow(plotter.screenshot())
        plt.title(f"{eig_name}: {eig_pairs[i][0]:.2f}")
        plt.axis("off")
        pdf.savefig(dpi=500)

        plt.clf()
        plotter.remove_actor(actor)

    pdf.close()

def eigs_to_pdf_3d(V, mesh, eig_pairs, data_label="u", eig_name="Eigenvalue", filename="output.pdf"):
    eig_pairs.sort(key=lambda x: abs(x[0]))

    pdf = pltpdf.PdfPages(filename)
    pdf = pltpdf.PdfPages(filename)
    plotter = pyvista.Plotter(off_screen=True, shape=(1, 3))
    plotter.store_image = True
    plotter.ren_win.SetSize([2500, 1500])

    topology, cell_types, geometry = plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    #- Export and import solution
    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    ncolors = 256
    plotter.add_axes(line_width=5)
    # cmap = plt.cm.get_cmap("PiYG")(range(ncolors))
    # cmap[:,-1] = np.concatenate([np.linspace(1.0, 0, ncolors//2), np.linspace(0, 1, ncolors//2)])
    # map_object = mplclr.LinearSegmentedColormap.from_list(name='mine',colors=cmap)
    # cmap=map_object

    p = fem.Function(V)
    for i in range(len(eig_pairs)):
        if np.isclose(eig_pairs[i][0].real, 0):
            continue
        p.vector[:] = eig_pairs[i][1]
        u_grid.point_data[data_label] = p.x.array.real
        u_grid.set_active_scalars(data_label)
        u_slices = pyvista.MultiBlock()
        # u_slices.append(u_grid.slice(normal=(-1, 1, 1), origin=grid.center))
        # u_slices.append(u_grid.slice(normal=(1, 1, 1), origin=grid.center))
        u_slices.append(u_grid.slice_along_axis(n=5, axis="x"))
        u_slices.append(u_grid.slice_along_axis(n=5, axis="y"))

        max_u = np.max(np.abs(p.x.array.real))
        outline = grid.outline()
        max_u = np.max(np.abs(p.vector))
        actors = []
        for j in range(3):
            plotter.subplot(0, j)
            
            plotter.add_mesh(outline)

            plotter.camera_position = 'yz'
            plotter.camera.Azimuth(60*j + 30)
            plotter.camera.Elevation(30)

            actors.append(plotter.add_mesh(u_slices, show_edges=False, opacity=1, clim=[-max_u, max_u]))

        plt.imshow(plotter.screenshot())
        plt.title(f"{eig_name}: {eig_pairs[i][0]:.2f}")
        plt.axis("off")
        pdf.savefig(dpi=500)

        plt.clf()
        for actor in actors:
            plotter.remove_actor(actor)

    pdf.close()


def eigs_to_pdf_1d_robin(x, eig_pairs, xlabel="x", ylabel="y", eig_name="Eigenvalue", filename="output.pdf"):
    eig_pairs.sort()
    pdf = pltpdf.PdfPages(filename)
    for i in range(len(eig_pairs)):
        p = np.real(eig_pairs[i][1])
        p_act = (np.sin(np.pi*i*x) + i*np.pi*np.cos(i*np.pi*x))
        if np.max(np.abs(p_act)) != 0:
            p_act = p_act*(np.max(np.abs(p))/np.max(np.abs(p_act)))
            ind = np.argmax(p)
            if np.sign(p_act[ind]) != np.sign(p[ind]):
                p_act = -p_act
        plt.plot(x, p, label="Numerical")
        plt.plot(x, p_act, label="Analytical")
        plt.xlabel(xlabel)
        plt.legend()
        plt.ylabel(ylabel)
        plt.title(f"{eig_name}: {eig_pairs[i][0]:.2f}")
        pdf.savefig()
        plt.clf()
    pdf.close()

def eigs_to_pdf_2d_robin(V, mesh, eig_pairs, data_label="u", eig_name="Eigenvalue", filename="output.pdf"):
    eig_pairs.sort()

    pdf = pltpdf.PdfPages(filename)
    plotter = pyvista.Plotter(off_screen=True)
    plotter.store_image = True
    plotter.ren_win.SetSize([2500, 1500])
    plt.axis("off")

    topology, cell_types, geometry = plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid, show_edges=True)

    #- Export and import solution
    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    plotter.add_axes(line_width=5)
    cmap = plt.cm.get_cmap("PiYG")

    p = fem.Function(V)
    n=0.5
    for i in range(len(eig_pairs)):
        p.vector[:] = np.real(eig_pairs[i][1])
        u_grid.point_data[data_label] = p.x.array.real
        u_grid.set_active_scalars(data_label)

        max_u = np.max(np.abs(p.vector))
        actor = plotter.add_mesh(u_grid, show_edges=True, clim=[-max_u, max_u], cmap=cmap)
        plotter.view_xy()
        plt.imshow(plotter.screenshot())
        plt.title(f"{eig_name}: {eig_pairs[i][0]:.2f}")
        plt.axis("off")
        pdf.savefig(dpi=500)

        plt.clf()
        plotter.remove_actor(actor)
        if np.isclose(eig_pairs[i][0].real, n, 0.001):
            x_line = np.linspace(0, 1, 100)
            p_sim = u_grid.sample_over_line((0, 0.5, 0), (1, 0.5, 0), resolution=99).point_data[data_label]
            p_act = (np.sin(np.pi*n*2*x_line) + n*2*np.pi*np.cos(n*2*np.pi*x_line))
            if np.max(np.abs(p_act)) != 0:
                p_act = p_act*(np.max(np.abs(p_sim))/np.max(np.abs(p_act)))
                ind = np.argmax(p_sim)
                if np.sign(p_act[ind]) != np.sign(p_sim[ind]):
                    p_act = -p_act  
            plt.plot(x_line, p_act, label="Analytical")
            plt.plot(x_line, p_sim, label="Numerical")
            plt.xlabel("x")
            plt.ylabel(data_label)
            plt.legend()
            pdf.savefig(dpi=500)
            plt.clf()
            n+=0.5
            continue

    pdf.close()

def eigs_to_pdf_3d(V, mesh, eig_pairs, data_label="u", eig_name="Eigenvalue", filename="output.pdf"):
    eig_pairs.sort()

    pdf = pltpdf.PdfPages(filename)
    pdf = pltpdf.PdfPages(filename)
    plotter = pyvista.Plotter(off_screen=True)
    plotter.store_image = True
    plotter.ren_win.SetSize([2500, 1500])

    topology, cell_types, geometry = plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    #- Export and import solution
    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    ncolors = 256
    plotter.add_axes(line_width=5)
    # cmap = plt.cm.get_cmap("PiYG")(range(ncolors))
    # cmap[:,-1] = np.concatenate([np.linspace(1.0, 0, ncolors//2), np.linspace(0, 1, ncolors//2)])
    # map_object = mplclr.LinearSegmentedColormap.from_list(name='mine',colors=cmap)
    # cmap=map_object

    p = fem.Function(V)
    n=0.5
    for i in range(len(eig_pairs)):
        if np.isclose(eig_pairs[i][0].real, 0):
            continue
        p.vector[:] = eig_pairs[i][1]
        u_grid.point_data[data_label] = p.x.array.real
        u_grid.set_active_scalars(data_label)
        # u_slices = pyvista.MultiBlock()
        # u_slices.append(u_grid.slice(normal=(-1, 1, 1), origin=grid.center))
        # u_slices.append(u_grid.slice(normal=(1, 1, 1), origin=grid.center))
        # u_slices.append(u_grid.slice_along_axis(n=5, axis="x"))
        # u_slices.append(u_grid.slice_along_axis(n=5, axis="y"))

        max_u = np.max(np.abs(p.x.array.real))
        outline = grid.outline()
        max_u = np.max(np.abs(p.vector))
        actors = []
        for j in range(1):
            
            plotter.add_mesh(outline)

            plotter.camera_position = 'yz'
            plotter.camera.Azimuth(60*j + 30)
            plotter.camera.Elevation(30)

            actors.append(plotter.add_mesh(u_grid, show_edges=False, opacity=1))

        plt.imshow(plotter.screenshot())
        plt.title(f"{eig_name}: {eig_pairs[i][0]:.2f}")
        plt.axis("off")
        pdf.savefig(dpi=500)

        plt.clf()
        for actor in actors:
            plotter.remove_actor(actor)
        
        x_line = np.linspace(0, 1, 100)
        p_sim = u_grid.sample_over_line((0, 0.5, 0.99), (1, 0.5, 0.99), resolution=99).point_data[data_label]
        p_act = (np.sin(np.pi*n*2*x_line) + n*2*np.pi*np.cos(n*2*np.pi*x_line))
        if np.max(np.abs(p_act)) != 0:
            p_act = p_act*(np.max(np.abs(p_sim))/np.max(np.abs(p_act)))
            ind = np.argmax(p_sim)
            if np.sign(p_act[ind]) != np.sign(p_sim[ind]):
                p_act = -p_act 
        # plt.plot(x_line, np.flip(p_act), label="Analytical")
        plt.plot(x_line, p_sim, label="Numerical")
        plt.xlabel("x")
        plt.ylabel(data_label)
        plt.legend()
        pdf.savefig(dpi=500)
        plt.clf()
        n+=0.5
        continue

    pdf.close()


def plot_mesh(mesh, V=None, func = None):
    plotter = pyvista.Plotter()
    topology, cell_types, geometry = plot.create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid)

    if func != None:
        u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
        u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        u_grid.point_data["f"] = func.x.array.real
        u_grid.set_active_scalars("f")
        pass
        
    plotter.show()