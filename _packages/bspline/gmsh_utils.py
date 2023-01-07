import gmsh
import numpy as np
import sys

def add_bspline_points(factory, curve):
    """
    Adds the control points in a BSpline Curve to gmsh

    Parameters
    ------------------------
    factory: 
        Gmsh factory object in which the points will be added e.g gmsh.model.occ
    curve: BSplineCurve
        The bspline curve object containing the control points to add

    Returns
    -------------------------
    point_tags: list
        A list of the identifier tags of each of the control points in gmsh
    """
    point_tags = []
    point_dict = {}
    for point, lc_i in zip(curve.ctrl_points, curve.lc):
        if tuple(point) not in point_dict:
            tag = factory.addPoint(*point, lc_i)
            point_dict[tuple(point)] = tag
        else:
            tag = point_dict[tuple(point)]

        point_tags.append(tag)

    return point_tags


def create_bspline_curve(factory, curve, lc=2e-2):
    """
    Adds a bspline curve to gmsh given a bspline curve object. If the curve is not closed a line is drawn
    from the last to the first control point to close the curve

    Parameters
    ----------------------------
    factory: 
        Gmsh factory object in which the points will be added e.g gmsh.model.occ
    curve: BSplineCurve
        The bspline curve object containing the control points to add
    lc: float
        The size of the mesh cells next to the curve. By default the lc of the curve object is used. If
        none is found then the lc of the curve is set to this parameterfor all control points and used (default: 2e-2)

    Returns
    ----------------------------
    curve_tag: int
        The identifier tag for the curve in gmsh
    point_tags: list of int
        A list of the identifier tags of each of the control points in gmsh

    """
    if curve.lc == None:
        curve.set_uniform_lc(lc)

    point_tags = add_bspline_points(factory, curve)

    C1 = factory.add_bspline(
        point_tags,
        -1, 
        degree = curve.degree,
        weights = curve.weights,
        knots = curve.knots,
        multiplicities = curve.multiplicities
    )

    if point_tags[0] == point_tags[-1]:
        curve_tag = gmsh.model.occ.addCurveLoop([C1])
    else:
        C2 = gmsh.model.occ.add_line(point_tags[0], point_tags[-1])
        curve_tag = gmsh.model.occ.addCurveLoop([C1, -C2])
        
    return curve_tag, point_tags


def create_bspline_curve_mesh(curves, lc=2e-2, show_mesh=False):
    """
    Given a list of bpsline curve objects, adds them to gmsh, adds a plane surface between them and creates a 2D mesh.
    The function returns the gmsh instance containing the mesh.

    Parameters
    --------------------------
    curves: list of BSplineCurve
        A list of the bspline curve objects defining the curves you want to add to gmsh
    lc: float
        The size of the mesh cells next to the curve. By default the lc of the curve object is used. If
        none is found then the lc of the curve is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    --------------------------
    gmsh:
        The gmsh object containing the mesh
    
    """
    if not isinstance(curves, list):
        curves = [curves]
        
    gmsh.initialize()
    factory = gmsh.model.occ
    
    curve_tags = []
    for curve in curves:
        curve_tag, _ = create_bspline_curve(factory, curve, lc)
        curve_tags.append(curve_tag)
    
    factory.add_plane_surface(curve_tags, 1)

    factory.synchronize()
    gmsh.model.mesh.generate(2)

    if show_mesh and '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    return gmsh


def create_bspline_curve_mesh_file(curves, filepath, lc=2e-2, show_mesh=False):
    """
    Given a list of bpsline curve objects, adds them to gmsh, adds a plane surface between them and creates a 2D mesh.
    The mesh is then written to the given filepath

    Parameters
    --------------------------
    curves: list of BSplineCurve
        A list of the bspline curve objects defining the curves you want to add to gmsh
    filepath: string
        The filepath you want to store the .msh file including the filename and extension at the end.
    lc: float
        The size of the mesh cells next to the curve. By default the lc of the curve object is used. If
        none is found then the lc of the curve is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)
    """
    gmsh = create_bspline_curve_mesh(curves, lc=lc, show_mesh=show_mesh)
    gmsh.write(filepath)


def create_bspline_curve_mesh_model(curves, lc=2e-2, show_mesh=False):
    """
    Given a list of bpsline curve objects, adds them to gmsh, adds a plane surface between them and creates a 2D mesh.
    The gmsh model instance is then returned

    Parameters
    --------------------------
    curves: list of BSplineCurve
        A list of the bspline curve objects defining the curves you want to add to gmsh
    filepath: string
        The filepath you want to store the .msh file including the filename and extension at the end.
    lc: float
        The size of the mesh cells next to the curve. By default the lc of the curve object is used. If
        none is found then the lc of the curve is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    ---------------------------
    gmsh.model:
        The gmsh model instance containing the msh
    """

    gmsh = create_bspline_curve_mesh(curves, lc=lc, show_mesh=show_mesh)
    return gmsh.model


def create_bspline_curve_plane(factory, curve, lc=2e-2):
    curve_tag, point_tags = create_bspline_curve(factory, curve, lc)
    surface_tag = gmsh.model.occ.add_plane_surface([curve_tag], 1)
    return surface_tag, point_tags


def create_bspline_surface(factory, surface, lc=2e-2):
    """
    Adds a bspline surface to gmsh given a bspline surface object.

    Parameters
    ----------------------------
    factory: 
        Gmsh factory object in which the points will be added e.g gmsh.model.occ
    surface: BSplineSurface
        The bspline surface object containing the control points to add
    lc: float
        The size of the mesh cells next to the curve. By default the lc of the curve object is used. If
        none is found then the lc of the curve is set to this parameterfor all control points and used (default: 2e-2)

    Returns
    ----------------------------
    surface_tag: int
        The identifier tag for the surface in gmsh
    point_tags: list of int
        A list of the identifier tags of each of the control points in gmsh
    """
    if surface.lc == None:
        surface.set_uniform_lc(lc)

    point_tags = add_bspline_points(factory, surface)

    surface_tag = factory.add_bspline_surface(
        point_tags,
        surface.nU,
        degreeU = surface.degreeU,
        degreeV = surface.degreeV,
        weights = surface.weights, 
        knotsU = surface.knotsU,
        knotsV = surface.knotsV,
        multiplicitiesU = surface.multiplicitiesU,
        multiplicitiesV = surface.multiplicitiesV
    )

    return surface_tag, point_tags


def create_bspline_surface_mesh(surfaces, lc=2e-2, show_mesh=False):
    """
    Given a list of bpsline surface objects, adds them to gmsh and creates a 2D mesh.
    The function returns the gmsh instance containing the mesh.

    Parameters
    --------------------------
    surfaces: list of BSplineSurface
        A list of the bspline surface objects defining the surface you want to add to gmsh
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    --------------------------
    gmsh:
        The gmsh object containing the mesh
    
    """
    if not isinstance(surfaces, list):
        surfaces = [surfaces]

    gmsh.initialize()
    factory = gmsh.model.occ

    surface_tags = []
    for surface in surfaces:
        surface_tag, _ = create_bspline_surface(factory, surface, lc)
        surface_tags.append(surface_tag)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    if show_mesh and '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    return gmsh


def create_bspline_surface_mesh_file(surfaces, filepath, lc=2e-2, show_mesh=False):
    """
    Given a list of bpsline surface objects, adds them to gmsh and creates a 2D mesh.
    The mesh is then written to the given filepath.

    Parameters
    --------------------------
    surfaces: list of BSplineSurface
        A list of the bspline surface objects defining the surface you want to add to gmsh
    filepath: string
        The filepath you want to store the .msh file including the filename and extension at the end.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)
    
    """
    gmsh = create_bspline_surface_mesh(surfaces, lc=lc, show_mesh=show_mesh)
    gmsh.write(filepath)


def create_bspline_surface_mesh_model(surfaces, lc=2e-2, show_mesh=False):
    """
    Given a list of bpsline surface objects, adds them to gmsh and creates a 2D mesh.
    The gmsh model instance containing the msh is returned.

    Parameters
    --------------------------
    surfaces: list of BSplineSurface
        A list of the bspline surface objects defining the surface you want to add to gmsh
    filepath: string
        The filepath you want to store the .msh file including the filename and extension at the end.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    ----------------------------
    gmsh.model:
        The gmsh model instance containing the msh
    
    """
    gmsh = create_bspline_surface_mesh(surfaces, lc=lc, show_mesh=show_mesh)
    return gmsh.model


def create_end_bspline_Ucurves(factory, surface, point_tags):
    """
    Given a bspline surface, finds the curves definining v = v_0 and v = v_m and
    adds them to gmsh.

    Parameters
    ------------------------------
    factory: 
        Gmsh factory object in which the points will be added e.g gmsh.model.occ
    surface: BSplineSurface
        The bspline surface object defining the bspline surface used. Lines of constant v must be
        closed for the surface
    point_tags: list of int
        A list of the gmsh tags for each of the control points for the bspline surface

    Returns
    ------------------------------
    curve_tags: list of int
        A list containing the gmsh tags for both of the curves. These are the tags for defining a curveloop.

    """
    nU = surface.nU
    nV = surface.nV

    W1 = factory.add_bspline(
        point_tags[0:nU], 
        degree = surface.degreeU,
        weights = surface.weights[0:nU],
        knots = surface.knotsU,
        multiplicities = surface.multiplicitiesU
    )
    C1 = factory.add_curve_loop([W1])

    W2 = factory.add_bspline(
        point_tags[(nV-1)*nU:len(point_tags)], 
        degree = surface.degreeU,
        weights = surface.weights[(nV-1)*nU:len(point_tags)],
        knots = surface.knotsU,
        multiplicities = surface.multiplicitiesU
    )
    C2 = factory.add_curve_loop([W2])

    return [C1, C2]


def create_bspline_volume_surfaces(factory, surfaces, lc=2e-2):
    """
    Given a list of bspline surface objects, adds the surfaces to gsmh. Lines of constant v must be
    closed for each surface


    Parameters
    ------------------------------
    factory: 
        Gmsh factory object in which the points will be added e.g gmsh.model.occ
    surfaces: list of BSplineSurface
        A list of the bspline surface objects defining the surface you want to add to gmsh. Lines of constant v must be
        closed for each surface.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)

    Returns
    ------------------------------
    gmsh_tags: list of lists
        A list containing 3 lists. The first being the gmsh tags for the surfaces. The second being the gmsh tags for the 
        curves defining u = u_0 for each of the curves and lastly a list containing the gmsh tags for the curves defining u = u_m
    """
    surface_tags = []
    start_curves = []
    end_curves = []
    for surface in surfaces:
        surface_tag, point_tags = create_bspline_surface(factory, surface, lc)
        surface_tags.append(surface_tag)
        
        if not surface.is_closed(): 
            raise ValueError("The surface must be fully closed except at its end faces")

        C1, C2 = create_end_bspline_Ucurves(factory, surface, point_tags)
        start_curves.append(C1)
        end_curves.append(C2)

    return [surface_tags, start_curves, end_curves]


def create_bspline_volume(factory, surfaces, lc=2e-2):
    """
    Given a list of bspline surface objects that are all closed along all lines of contant v, the surfaces are added to gmsh and closed
    at its end faces using a planar surface to create a volume.

    Parameters
    ------------------------------
    factory: 
        Gmsh factory object in which the points will be added e.g gmsh.model.occ
    surface: list of BSplineSurface
        A bspline surface object defining the surface you want to add to gmsh. Lines of constant v must be
        closed for the surface.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)

    Returns
    ------------------------------
    V1: int
        The gmsh tag for the volume

    """
    surface_tags, start_curves, end_curves = create_bspline_volume_surfaces(factory, surfaces, lc=lc)

    S1 = factory.add_plane_surface(start_curves)
    S2 = factory.add_plane_surface(end_curves)

    Sl_1 = factory.add_surface_loop([S1, surface_tags[0], S2])
    V1 = factory.add_volume([Sl_1])

    return V1


def create_bspline_volume_endbsplines(factory, surfaces, start_surface, end_surface, lc=2e-2):
    """
    Given a list of bspline surface objects that are all closed along all lines of contant v, the surfaces are added to gmsh and closed
    at its end faces using two passed in bspline surfaces to create a volume.

    Parameters
    ------------------------------
    surfaces: list of BSplineSurface
        A list of bspline surface objects defining the surfaces you want to add to gmsh. Lines of constant v must be
        closed for the surface.
    start_surface: BSplineSurface
        The bspline surface object defining the surface used to close the v = v_0 side of the volume
    end_surface: BSplineSurface
        The bspline surface object defining the surface used to close the v = v_m side of the volume
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    ------------------------------
    V1:
        The gmsh tag for the volume

    """
    surface_tags, start_curves, end_curves = create_bspline_volume_surfaces(factory, surfaces, lc=lc)

    S1 = create_bspline_surface(start_surface)
    S2 = create_bspline_surface(end_surface)

    Sl_1 = factory.add_surface_loop([S1, surface_tags[0], S2])
    V1 = factory.add_volume([Sl_1])

    return V1


def create_bspline_volume_mesh(surfaces, lc=2e-2, show_mesh=False):
    """
    Given a list of bspline surface objects that are all closed along all lines of contant v, the surfaces are added to gmsh and closed
    at its end faces using a planar surface to create a volume.

    Parameters
    ------------------------------
    surfaces: list of BSplineSurface
        A list of bspline surface objects defining the surfaces you want to add to gmsh. Lines of constant v must be
        closed for the surface.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    ------------------------------
    gmsh:
        The gmsh object containing the mesh

    """
    if not isinstance(surfaces, list):
        surfaces = [surfaces]

    gmsh.initialize()
    factory = gmsh.model.occ

    volume_tag = create_bspline_volume(factory, surfaces, lc)
    gmsh.model.occ.heal_shapes()
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    if show_mesh and '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    return gmsh


def create_bspline_volume_mesh_file(surface, filepath, lc=2e-2, show_mesh=False):
    """
    Given a list of bspline surface objects that are all closed along all lines of contant v, the surfaces are added to gmsh and closed
    at its end faces using a planar surface to create a volume. The mesh is written to a .msh file at the file path given.

    Parameters
    ------------------------------
    filepath: string
        The filepath you want to store the .msh file including the filename and extension at the end.
    surfaces: list of BSplineSurface
        A list of bspline surface objects defining the surfaces you want to add to gmsh. Lines of constant v must be
        closed for the surface.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    ------------------------------
    gmsh:
        The gmsh object containing the mesh

    """
    gmsh = create_bspline_volume_mesh(surface, lc=lc, show_mesh=show_mesh)
    gmsh.write(filepath)


def create_bspline_volume_mesh_model(surface, lc=2e-2, show_mesh=False):
    """
    Given a list of bspline surface objects that are all closed along all lines of contant v, the surfaces are added to gmsh and closed
    at its end faces using a planar surface to create a volume. The gmsh model instance is returned.

    Parameters
    ------------------------------
    surfaces: list of BSplineSurface
        A list of bspline surface objects defining the surfaces you want to add to gmsh. Lines of constant v must be
        closed for the surface.
    lc: float
        The size of the mesh cells next to the surface. By default the lc of the surface object is used. If
        none is found then the lc of the surface is set to this parameterfor all control points and used (default: 2e-2)
    show_mesh: bool
        True if you want the gmsh window to open showing the mesh after the function is ran else false 
        (default: False)

    Returns
    ----------------------------
    gmsh.model:
        The gmsh model instance containing the msh

    """
    gmsh = create_bspline_volume_mesh(surface, lc=lc, show_mesh=show_mesh)
    return gmsh.model

