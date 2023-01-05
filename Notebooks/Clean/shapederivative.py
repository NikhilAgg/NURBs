import numpy as np
import ufl
from dolfinx import fem, plot
from dolfinx_utils2 import conjugate_function, quadratic_product
import pyvista
import matplotlib.pyplot as plt

def plotp(p, msh, degree, boundary=1, mesh=False, ret=False):
    point = {
        1: [(0, 0, 0), (0.99999, 0, 0)],
        2: [(0, 0, 0), (0, 0.999999, 0)],
        3: [(0, 0.999999, 0), (1, 0.999999, 0)],
        4: [(0.99999999, 0, 0), (0.9999999, 0.99999999, 0)]
    }
    V = fem.FunctionSpace(msh, ("Lagrange", degree))
    plotter = pyvista.Plotter()
    
    topology, cell_types, geometry = plot.create_vtk_mesh(msh, msh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter.add_mesh(grid, show_edges=True)

    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

    u_grid.point_data["p"] = p.x.array.real
    
    u_grid.point_data["pc"] = p.x.array.imag
    u_grid.set_active_scalars("p")
    plotter.add_mesh(u_grid, show_edges=True)
    x_line = np.linspace(0, 1, 100)
    p_sim = u_grid.sample_over_line(*point[boundary], resolution=99).point_data["p"]
    pc_sim = u_grid.sample_over_line(*point[boundary], resolution=99).point_data["pc"]
    # p_act = np.sqrt(2)*np.pi*np.sin(np.pi*x_line)**2
    # p_act = -np.pi*np.sin(np.pi*x_line)
    # plt.plot(x_line, p_act, label="Analytical")
    if not mesh:
        plt.plot(x_line, p_sim)
        # plt.plot(x_line, p_act, label="Analytical")
        plt.legend()
    else:
        plotter.view_xy()
        plotter.show()
    if ret:
        return p_sim + pc_sim*1j

class ShapeDerivative:
    def __init__(self, model):
        self.model = model
        self.c = self.model.params['c']
        
    def get_dL_dw(self, omega):
        B = self.model.matrices.B if self.model.isB else None
        C = self.model.matrices.C
        D = self.model.D if self.model.isD else None

        dL_dw = (2*omega)*C

        if self.model.isB:
            dL_dw += B
        
        if self.model.isD:
            dL_dw += D.get_derivative(omega)

        return dL_dw

    def normalise_adjoint(self, omega, p, p_adj):
        p_conj = conjugate_function(p_adj, self.model.V)
        dL_dw = self.get_dL_dw(omega)

        first_term = quadratic_product(p_conj, dL_dw, p, self.model.V)

        if self.model.isB:
            second_term = quadratic_product(p_conj, self.model.matrices.B, p, self.model.V)
        else:
            second_term = 0
        
        norm_const = first_term + second_term

        p_adj.vector[:] = p_adj.vector[:]/np.conjugate(norm_const)

    def calculate_G(self, p, p_adj, kappa=0):
        p_conj = conjugate_function(p_adj, self.model.V) 
        
        if all("Dirichlet" == cond_type for cond_type in self.model.bound_cond_types):
            G = -self.c**2*ufl.Dn(p)*ufl.Dn(p_conj)
        elif all("Neumann" == cond_type for cond_type in self.model.bound_cond_types):
            G = ufl.div(p_conj*(self.c**2)*ufl.grad(p))
        else:
            G = - p_conj * (kappa*(self.c**2)+ self.c*ufl.Dn(self.c))*ufl.Dn(p) + ufl.div(p_conj*(self.c**2)*ufl.grad(p)) - 2*ufl.Dn(p_conj)*(self.c**2)*ufl.Dn(p)

        return G

    def calculate_shape_derivative(self, omega, p, p_adj, C, boundary, kappa=0):
        self.normalise_adjoint(omega, p, p_adj)
        G = self.calculate_G(p, p_adj, kappa)
        dw = fem.assemble_scalar(fem.form(C * G * self.model.ds(boundary)))

        return dw

    def perform_taylor_test(self, C, boundary, new_mesh_func, new_params_func, step, target, nev, i, it=5, start=0):
        omega, p, p_adj = self.model.find_eigenvalue(target, nev, i)
        dw = self.calculate_shape_derivative(omega, p, p_adj, C, boundary)
        print(dw)

        x_points = [0]
        y_points = [0]

        epsilon = start
        for _ in range(1, it):
            epsilon += step

            new_msh, new_facet_tags = new_mesh_func(epsilon)
            self.model.update_mesh(new_msh, new_facet_tags)
            self.params = new_params_func(new_msh, self.model.V)
            omega_new = self.model.find_eigenvalue(target, nev, i)[0]

            Delta_w_FD = omega_new - omega
            x_points.append(epsilon**2)
            y_points.append(abs(Delta_w_FD - dw*epsilon))

        return [x_points, y_points]
