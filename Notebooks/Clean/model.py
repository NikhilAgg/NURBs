import numpy as np
from dolfinx import fem, plot
from ufl import Measure
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.active_flame_x import ActiveFlameNT4
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep, fixed_point_iteration_eps, eps_solver, pep_solver
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.parameters_utils import rho, gaussianFunction
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

def param_dict(c, w, h, rho, tau, Q_tot, U_bulk, eta, isFlame):
    params = {}
    params['c'] = c
    params['w'] = w
    params['h'] = h
    params['rho'] = rho
    params['tau'] = tau
    params['Q_tot'] = Q_tot
    params['U_bulk'] = U_bulk
    params['eta'] = eta
    params['isFlame'] = isFlame

    return params

class Model:
    def __init__(self, msh, facet_tags, degree, bound_cond, params, cell_type="Lagrange"):
        
        self.msh = msh
        self.facet_tags = facet_tags
        self.degree = degree
        self.params = params
        self.cell_type = cell_type
        self.subdomains = []

        self.bound_cond = bound_cond
        self.bound_cond_types = [list(cond.keys())[0] for cond in bound_cond.values()]
        
        self.V = fem.FunctionSpace(msh, (cell_type, degree))
        self.ds = Measure('ds', domain=msh, subdomain_data=facet_tags)

        self.isB = True if 'Robin' in self.bound_cond_types else False
        self.isD = True if self.params['isFlame'] else False
        return
        

    def construct_passive_flame_matrices(self):
        self.matrices = PassiveFlame(self.msh, self.facet_tags, self.bound_cond, self.params['c'] , degree = self.degree)

        self.matrices.assemble_A()

        if self.isB:
            self.matrices.assemble_B()

        self.matrices.assemble_C()


    def construct_active_flame_matrices(self):
        if not self.isD:
            return

        self.D = ActiveFlameNT4(self.msh, 
                                self.subdomains, 
                                self.params['w'], 
                                self.params['h'], 
                                self.params['rho'], 
                                self.params['Q_tot'], 
                                self.params['U_bulk'], 
                                self.params['eta'], 
                                self.params['tau'], 
                                degree=self.degree)   


    def solveEigenvalueProblem(self, target, nev=2, i=0):
        if self.isB and self.isD:
            E = fixed_point_iteration_pep(self.matrices, self.D, target, i=i, print_results= False)
        elif self.isB and not self.isD:
            E = pep_solver(self.matrices.A, self.matrices.B, self.matrices.C, target, nev = nev)
        elif not self.isB and self.isD:
            E = fixed_point_iteration_eps(self.matrices, self.D, target, i=i, print_results= False, two_sided=True)
        else:
            E = eps_solver(self.matrices.A, self.matrices.C, target, nev = nev, two_sided=True)

        return E


    def find_eigenvalue(self, target, nev, i):
        self.construct_passive_flame_matrices()
        self.construct_active_flame_matrices()

        E = self.solveEigenvalueProblem(target, nev, i)

        omega, p = normalize_eigenvector(self.msh, E, i, degree=self.degree, which='right')
        omega_adj, p_adj = normalize_eigenvector(self.msh, E, i, degree=self.degree, which='left')

        return [omega, p, p_adj]

    
    def update_mesh(self, msh, facet_tags, subdomains=[]):
        self.msh = msh 
        self.facet_tags = facet_tags
        self.subdomains = subdomains

        self.V = fem.FunctionSpace(msh, (self.cell_type, self.degree))
        self.ds = Measure('ds', domain=msh, subdomain_data=facet_tags)
