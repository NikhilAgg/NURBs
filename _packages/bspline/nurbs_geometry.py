"""A module containing classes for creation of meshes in gmsh using NURBs defined in NURBsCurve and NURBsSurface instances
and also allows for the import of those meshes in fenics along with helping functions."""

import gmsh
import sys
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx import fem

class NURBsGeometry:
    def __init__(self, nurbs = []):
        """
        Define the nurbs that make up your geometry. The nurbs will be stored in self.nurbs

        Parameters
        ---------------------------------
        nurbs: 2d list of NURBsCurves or NURBsSurface
            A 2d list where each row contains the nurbs that form a closed surface. The first row defines the outer
            surface of the geometry. Subsequent rows define the closed surfaces that are subtracted from the geometry.
        """
        self.nurbs = nurbs

        self.nurbs_tags = []

        self.nurbs_groups = []

        self.point_dict = {}

        #Gmsh
        self.gmsh = gmsh
        self.gmsh.initialize() if not self.gmsh.isInitialized() else None
        self.gmsh.option.setNumber("General.Terminal", 0)
        self.model = self.gmsh.model
        self.factory = self.model.occ
        self.node_to_params = {}

        #Fenics
        self.V = None
        self.boundary_conditions = None


    def add_nurbs_points(self, nurb):
        """
        Given a nurbs object, this function will add each of its control points to gmsh.

        Parameters
        ----------------------------
        nurb: NURBsCurve or NURBsSurface
            The nurbs whose control points will be added to gmsh
        """
        point_tags = []

        ctrl_points = nurb.ctrl_points.reshape(-1, nurb.dim)
        ctrl_points = np.c_[ctrl_points, np.zeros((ctrl_points.shape[0], np.abs(3-self.dim)))] #Makes the ctrl points at least 3D so that they can be passed into gmsh
        for point, lc_i in zip(ctrl_points, nurb.lc.flatten()):
            if tuple(point) not in self.point_dict:
                tag = self.factory.addPoint(*point, lc_i)
                self.point_dict[tuple(point)] = tag
            else:
                tag = self.point_dict[tuple(point)]

            point_tags.append(tag)

        return point_tags


    def add_nurbs_groups(self, nurbs_groups, group_names=None):
        """
        Assign physical groups within gmsh to the nurbs that make up your geometry. The physical groups will be stored
        in self.nurbs_groups.

        Parameters
        ----------------------------
        nurbs_groups: list of list of ints
            The tags for the physical group you want to assign to each nurb defined in self.nurbs. These argument
            should have the same shape as self.nurbs where each element is the tag for the corresponding nurb
        group_names: dict
            A dictionary that maps the physical group tags to their names. (Default: None)
        """
        self.nurbs_groups = nurbs_groups

        if self.nurbs_groups != []:
            for i, row in enumerate(nurbs_groups):
                for j, group in enumerate(row):
                    self.model.addPhysicalGroup(self.dim-1, [self.nurbs_tags[i][1][j]], group)

        if group_names:
            for tag, name in group_names:
                self.model.setPhysicalName(self.dim-1, tag, name)
                

    def model_to_fenics(self, comm=MPI.COMM_WORLD, rank=0, show_mesh=False):
        """
        Generates the mesh for the model in self.model and then imports it from gmsh into fenics. The mesh is then 
        stored in self.msh, cell_markers in self.cell_markers and facet_tags in self.facet_tags.

        Parameters
        ----------------------------
        comm: MPI Communicator
            (Default: MPI.COMM_WORLD)
        rank: int
            Number of process (Default: 0)
        show_mesh: bool
            If true, a gmsh window will open showing the mesh that has been created. (Default: False)
        """
        self.model.mesh.generate(self.dim)
        self.msh, self.cell_markers, self.facet_tags = gmshio.model_to_mesh(self.model, comm, rank)

        t_imap = self.msh.topology.index_map(self.msh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        total_num_cells = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM) #sum all cells and distribute to each process
        if MPI.COMM_WORLD.Get_rank()==0:
            print("\nNumber of cells:  {:,}".format(total_num_cells))
            print("Number of cores: ", MPI.COMM_WORLD.Get_size(), "\n")

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


    def create_function_space(self, cell_type, degree):
        """
        Given a cell type and degree, defines a function space for the geometry. This function space
        is then stored in self.V for later access.

        Parameters
        ----------------------------
        cell_type: string
            The name of the cell type used in the finite element
        degree: int
            The degree of the basis functions used in the finite element.
        """
        if not self.msh:
            print("Geometry has no fenics msh. Run model_to_fenics to create a msh")
            return

        self.cell_type = cell_type
        self.degree = degree
        self.V = fem.FunctionSpace(self.msh, (cell_type, degree))

        return self.V


    def get_displacement_field(self, typ, nurb_ind, param_ind, flip_norm=None, tie=True, cell_type=None, degree=None):
        """
        Given the index of a nurb parameter generates the displacement field for that parameter where the displacment field is the 
        derivative of the mesh wrt to that parameter in the direction normal to the point. The returned function will use the function
        space stored in self.V unless both degree and cell_type are set.

        Parameters
        ----------------------------
        typ: string
            Either "control point" or "weight" depending on which parameter you want to find the displacement field wrt.
        nurb_ind: tuple of int
            The indicies of the nurb (in self.nurbs) whose parameter you want to edit.
        param_ind: tuple of int
            The indicies of the parameter which the displacement field is calculated wrt. If the nurb is a NURBsCurve the tuple should only 
            contain 1 value corresponding to the index of the parameter
        flip_norm: bool
            If true will consider the displacement in the direction of the opposite unit normal (since each point will have 2 unit normals
            corresponding to different directions). Note that this is equivalent to multiplying the result by -1. If None is passed in, 
            the value of flip in self.flip_norm for each nurb will be used (default: None)
        tie: bool
            If true, then the displacement field at a point on the mesh will be equal to the sum of the derivatives for all parameters across
            all the nurbs in self.nurbs that share the same coordinates as those as the parameter passed in. If typ = "weight", then it will be the sum
            of the derivatives of all parameters that share the same coordinates and weight. (Default: True)
        cell_type: string
            The name of the cell type used in the finite element and for which the returned displacement field will be defined in. If None
            then the function space in self.V is used (Default: None)
        degree: int
            The degree of the basis functions used in the finite element. (Default: None)

        Returns:
        -------------------------------------------
        C: fem.Function
            The displacement field wrt to the passed in parameter. The function will be defined either on the function space using the cell_type
            and degree passed in or otherwise the one in self.V.

        """
        deriv_nurbs = self.get_deriv_nurbs(typ, nurb_ind, param_ind, tie)

        if typ == "control point":
            dim = self.dim
        else:
            dim = 1
  
        def get_displacement(nurbs):
            nonlocal typ, flip_norm, deriv_nurbs, nurb_ind
            for indices, params in nurbs:
                C = np.zeros(dim)
                if tuple(indices) in deriv_nurbs: 
                    i, j = indices
                    param_ind = deriv_nurbs[tuple(indices)][0]
                    C += self.nurbs[i][j].get_displacement(typ, *params, *param_ind, flip_norm, tie)

            return C

        degree = degree or self.degree
        cell_type = cell_type or self.cell_type
        return self.function_from_params(dim, get_displacement, cell_type, degree)
    

    def get_curvature_field(self, nurb_ind=None, flip_norm=None, cell_type=None, degree=None):
        """
        Calculates the curvature field along the surface of geometry and stores it in a fenics Function. The returned function will use the function
        space stored in self.V unless both degree and cell_type are set. The curvature field can be calculated for just one nurb surface/curve or considering all
        nurbs for the entire curvature field around the boundary

        Parameters
        ----------------------------
        nurb_ind: tuple of int
            The indicies of the nurb (in self.nurbs) for which you want to calculate the curvature field. If the None, the curvature for the entire
            boundary is calculated. At nodes where two nurbs meet the curvature will be calculated wrt to one of them randomly. (Default: None)
        flip_norm: bool
            If true will consider the curvature as the divergence of the unit normal in the opposite direction (since each point will have 2 unit normals
            corresponding to different directions). Note that this is equivalent to multiplying the result by -1. If None is passed in, 
            the value of flip in self.flip_norm for each nurb will be used (default: None)
        cell_type: string
            The name of the cell type used in the finite element and for which the returned displacement field will be defined in. If None
            then the function space in self.V is used (Default: None)
        degree: int
            The degree of the basis functions used in the finite element. (Default: None)

        Returns:
        -------------------------------------------
        C: fem.Function
            The displacement field wrt to the passed in parameter. The function will be defined either on the function space using the cell_type
            and degree passed in or otherwise the one in self.V.

        """
        def get_curvature(nurbs):
            nonlocal flip_norm, nurb_ind
            for indices, params in nurbs:
                if not nurb_ind or tuple(indices) == tuple(nurb_ind): 
                    i, j = indices
                    k = self.nurbs[i][j].get_curvature(*params, flip_norm)

            return k

        degree = degree or self.degree
        cell_type = cell_type or self.cell_type
        return self.function_from_params(1, get_curvature, cell_type, degree)

    
    def function_from_params(self, dim, func, cell_type=None, degree=None):
        """
        Executes a passed in function "func" at each node on the surface of the mesh. The result of the function
        at each node is then stored in a Fenics function. The resulting Fenics function is then interpolated onto
        a different function space (either the one in self.V or the one corresponding to the arguements degree and cell type)

        Parameters
        ----------------------------
        dim: int
            The dimension of the return type of the function you are executing at each node.
        func: function
            The function which you want to execute at each node. The function must only take in one arguement which will be a list
            of lists each of which contains indicies to a nurb the node belongs too and its u and v values for that nurb. The function 
            must return an int (dim = 1), 1d list or 1d numpy array (dim = length of list/array)
        cell_type: string
            The name of the cell type used in the finite element and for which the returned displacement field will be defined in. If None
            then the function space in self.V is used (Default: None)
        degree: int
            The degree of the basis functions used in the finite element. (Default: None)

        Returns
        --------------------------------
        C: fem.Function
            The function containing the result of the function executed at each node. The function has been interpolated the function space
            specified or that in self.V.
        """
        degree = degree or self.degree
        cell_type = cell_type or self.cell_type

        if dim == 1:
            V = fem.FunctionSpace(self.msh, (cell_type, 1))
            V2 = fem.FunctionSpace(self.msh, (cell_type, degree))
        else:
            V = fem.VectorFunctionSpace(self.msh, (cell_type, 1), dim=dim)
            V2 = fem.VectorFunctionSpace(self.msh, (cell_type, degree), dim=dim)

        c = fem.Function(V)
        C = fem.Function(V2)

        def interp_func(x):
            nonlocal dim
            C = np.zeros((dim, len(x[0])), dtype=np.complex128)
            for k, coord in enumerate(zip(x[0], x[1], x[2])):
                coord = np.around(coord, self.node_map_decimal_points)
                
                if tuple(coord) in self.node_to_params:
                    C[:, k] = func(self.node_to_params[tuple(coord)])

            return C

        c.interpolate(interp_func)
        C.interpolate(c)

        return C

    
    def get_deriv_nurbs(self, typ, nurb_ind, param_ind, tie=True):
        """
        Given a nurb and a parameter index returns a dictionary where the keys are all the nurbs in self.nurbs
        that contain the same parameter and the values is a list of the indicies of the parameters in the nurb that match
        the one passed in. If typ = "weight", then the dictionary will include indicies of all parameters that share the same 
        coordinates and weight.

        Parameters
        -------------------------
        typ: string
            Either "control point" or "weight" depending on type of parameter you are referencing.
        nurb_int: tuple of int
            The indicies of the nurb (in self.nurbs) which contains the parameter.
        param_ind: tuple of int
            The indicies of the parameter. If the nurb is a NURBsCurve the index should still be in a tuple.
        tie: bool
            If false, then just the map {tuple(nurb_ind): [param_ind]} is returned. (Default: false)

        Returns
        --------------------------
        deriv_nurbs: dict
            The dictionary containing the nurbs and indicies described above
        """
        if not tie:
            return {tuple(nurb_ind): [param_ind]}

        deriv_nurbs = {}
        i, j = nurb_ind
        if len(param_ind) == 1:
            point = self.nurbs[i][j].ctrl_points[param_ind[0]]
            weight = self.nurbs[i][j].weights[param_ind[0]]
        elif len(param_ind) == 2:
            point = self.nurbs[i][j].ctrl_points[param_ind[0]][param_ind[1]]
            weight = self.nurbs[i][j].weights[param_ind[0]][param_ind[1]]

        for j, nurb in enumerate(self.nurbs[i]):
            if nurb.ctrl_points_dict == {}:
                nurb.create_ctrl_point_dict()

            if tuple(point) in nurb.ctrl_points_dict:
                param_inds = nurb.ctrl_points_dict[tuple(point)]
                if typ == "control point":
                    deriv_nurbs[(i, j)] =  param_inds
                elif typ == "weight":
                    deriv_params = []
                    for params in param_inds:
                        if np.all(nurb.weights[params[0]] == weight) or nurb.weights[params[0]][params[1]] == weight:
                            deriv_params.append(params)
                    
                    deriv_nurbs[(i, j)] = deriv_params

        return deriv_nurbs

    
    def create_node_to_param_map(self, decimal_points=8):
        """
        Creates a map from the coordinates of the nodes of the mesh to a list of lists containing the indicies of the nurbs
        in self.nurbs that share that point and the corresponding u and v value. The map is stored in self.node_to_params

        Parameters
        ---------------------------------
        decimal_points: int
            The number of decimal points you want the coordinates of the nodes to be stored as in the map
        """
        self.node_map_decimal_points = decimal_points
        for i, row in enumerate(self.nurbs_tags):
            for j, tag in enumerate(row[1]):
                nodes = self.model.mesh.getNodes(self.dim-1, tag)[1]
                params = self.model.getParametrization(self.dim-1, tag, nodes)
                p = self.dim-1
                for k in range(len(params)//p):
                    self.add_params_to_map(nodes[3*k:3*k+3], [i, j], params[p*k:p*k+p])
                

    def add_params_to_map(self, coord, indices, params):
        """
        Helper functions for create_node_to_param_map. Adds a node coordinate to self.node_to_params map along
        with indicies for a nurb that shares the node and its corresponding u/and v values.

        Parameters
        -----------------------------------
        coord: list or 1d numpy array
            Coordinate of node
        indicies: tuple
            Indicies of nurb in self.nurb that contains the node 
        params: tuple
            The u/and v values corresponding to the node for the nurb at indicies.
        """
        coord = np.around(coord, self.node_map_decimal_points)
        if tuple(coord) not in self.node_to_params:
            self.node_to_params[tuple(coord)] = [[indices, params]]
        else:
            for ind, _ in self.node_to_params[tuple(coord)]:
                if indices == ind:
                    return

            self.node_to_params[tuple(coord)].append([indices, params])


    def set_boundary_conditions(self, boundary_conditions):
        """
        Stores a given set of boundary conditions in the class in self.boundary_conditions.

        Parameters
        -------------------------------
        boundary_conditions:
            The boundary conditions
        """
        self.boundary_conditions = boundary_conditions


class NURBs2DGeometry(NURBsGeometry):
    def __init__(self, nurbs=[]):
        """
        Define the nurbs that make up your geometry. The nurbs will be stored in self.nurbs

        Parameters
        ---------------------------------
        nurbs: 2d list of NURBsCurves
            A 2d list where each row contains the nurb curves that form a closed loop. The first row defines the outer
            surface of the geometry. Subsequent rows define the closed surfaces that are subtracted from the geometry.
        """
        self.dim = 2
        super().__init__(nurbs)


    def create_nurb_curves(self, nurbs):
        """
        Given a list of nurb curves adds them to gmsh and creates a curveloop containing them. Note that the first
        and last nurb must meet.

        Parameters
        ---------------------------
        nurbs: list of NURBsCurve
            List of nurbs curves that you want to add to gmsh

        Returns
        --------------------------
        curveloop_tag: int
            The tag for the created curveloop in gmsh
        curve_tags: list of int
            The tags for each of the created nurb curves within gmsh
        """
        curve_tags = []
        for nurb in nurbs:
            point_tags = self.add_nurbs_points(nurb)

            multiplicities = nurb.get_periodic_multiplicities()

            curve_tag = self.factory.add_bspline(
                point_tags,
                -1, 
                degree = nurb.degree,
                weights = nurb.weights,
                knots = nurb.knots,
                multiplicities = multiplicities,
            )

            curve_tags.append(curve_tag)

        curveloop_tag = gmsh.model.occ.addCurveLoop(curve_tags)
            
        return [curveloop_tag, curve_tags]


    def generate_mesh(self, show_mesh=False):
        """
        Generates the gmsh model using the nurbs defined in self.nurbs.

        Parameters
        ---------------------------
        show_mesh: bool
            If true, a gmsh window will open showing the geometry that has been created. (Default: False)
        """
        curveloop_tags = []
        for nurb_list in self.nurbs:
            nurb_tags = self.create_nurb_curves(nurb_list)
            
            curveloop_tags.append(nurb_tags[0])
            self.nurbs_tags.append(nurb_tags)
        
        self.surface_tag = self.factory.add_plane_surface([*curveloop_tags])

        self.factory.synchronize()
        self.model.addPhysicalGroup(self.dim, [self.surface_tag], 1)
        self.gmsh.option.set_number("Mesh.MeshSizeFromParametricPoints", 1)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


class NURBs3DGeometry(NURBsGeometry):
    def __init__(self, nurbs=[]):
        """
        Define the nurbs that make up your geometry. The nurbs will be stored in self.nurbs

        Parameters
        ---------------------------------
        nurbs: 2d list of NURBsSurface
            A 2d list where each row contains the nurb surfaces that form a closed surface. The first row defines the outer
            surface of the geometry. Subsequent rows define the closed surfaces that are subtracted from the outer volume.
        """
        self.dim = 3
        super().__init__(nurbs)

        #Other
        self.ctrl_point_dict = {}

    
    def create_nurb_surfaces(self, nurbs):
        """
        Given a list of nurb surfaces adds them to gmsh and creates a surfaceloop containing them.

        Parameters
        ---------------------------
        nurbs: list of NURBsSurface
            List of nurbs surfaces that you want to add to gmsh

        Returns
        --------------------------
        surfaceloop_tag: int
            The tag for the created surfaceloop in gmsh
        surface_tags: list of int
            The tags for each of the created nurb surfaces within gmsh
        """
        surface_tags = []
        for nurb in nurbs:
            point_tags = self.add_nurbs_points(nurb)

            multiplicitiesU, multiplicitiesV = nurb.get_periodic_multiplicities()
            surface_tag = self.factory.add_bspline_surface(
                point_tags,
                nurb.nU,
                degreeU = nurb.degreeU,
                degreeV = nurb.degreeV,
                weights = nurb.weights.flatten(), 
                knotsU = nurb.knotsU,
                knotsV = nurb.knotsV,
                multiplicitiesU = multiplicitiesU,
                multiplicitiesV = multiplicitiesV
            )
            surface_tags.append(surface_tag)

        self.factory.heal_shapes(makeSolids=False)
        surfaceloop_tag = self.factory.add_surface_loop(surface_tags)

        return [surfaceloop_tag, surface_tags]


    def generate_mesh(self, show_mesh=False):
        """
        Generates the gmsh model using the nurbs defined in self.nurbs.

        Parameters
        ---------------------------
        show_mesh: bool
            If true, a gmsh window will open showing the geometry that has been created. (Default: False)
        """
        surfaceloop_tags = []
        for nurb_list in self.nurbs:
            nurb_tags = self.create_nurb_surfaces(nurb_list)
            
            surfaceloop_tags.append(nurb_tags[0])
            self.nurbs_tags.append(nurb_tags)
        
        self.volume_tag = self.factory.add_volume([*surfaceloop_tags], 1)
        self.factory.synchronize()
        self.model.addPhysicalGroup(self.dim, [self.volume_tag], 1)
        self.gmsh.option.set_number("Mesh.MeshSizeFromParametricPoints", 1)

        if show_mesh and '-nopopup' not in sys.argv:
            gmsh.fltk.run()


    def create_ctrl_point_dict(self):
        """
        Creates a dictionary that maps control points to a list of indicies of nurbs in self.nurbs
        that contain the control point. The dictionary is stored in self.ctrl_point_dict
        """
        for i, nurb_list in enumerate(self.nurbs):
            for j, nurb in enumerate(nurb_list):
                for point in nurb.ctrl_points.reshape(-1, nurb.dim):
                    if tuple(point) in self.ctrl_point_dict:
                        self.ctrl_point_dict[tuple(point)].add((i, j))
                    else:
                        self.ctrl_point_dict[tuple(point)] = set([(i, j)])

