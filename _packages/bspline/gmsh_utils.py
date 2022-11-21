import gmsh
import numpy as np
import sys
from bspline.bspline_curve import BSplineCurve

def add_bspline_points(factory, curve):
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


def create_bspline_curve_plane(factory, curve, lc=2e-2):
    curve_tag, point_tags = create_bspline_curve(factory, curve, lc)
    surface_tag = gmsh.model.occ.add_plane_surface([curve_tag], 1)
    return surface_tag, point_tags


def create_bspline_surface(factory, surface, lc=2e-2):

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


def create_end_bspline_Ucurves(factory, surface, point_tags):
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


def create_bspline_surface_volume(factory, surfaces, lc=2e-2):
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

    S1 = factory.add_plane_surface(start_curves)
    S2 = factory.add_plane_surface(end_curves)

    Sl_1 = factory.add_surface_loop([S1, surface_tags[0], S2])
    V1 = factory.add_volume([Sl_1])

    return [V1, point_tags]


def create_bspline_curve_mesh(curves, lc=2e-2, show_mesh=False):
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


def create_bspline_curve_mesh_file(curve, filepath, lc=2e-2, show_mesh=False):
    gmsh = create_bspline_curve_mesh(curve, lc=lc, show_mesh=show_mesh)
    gmsh.write(filepath)


def create_bspline_surface_mesh_model(curve, lc=2e-2, show_mesh=False):
    gmsh = create_bspline_curve_mesh(curve, lc=lc, show_mesh=show_mesh)
    return gmsh.model


def create_bspline_surface_mesh(surfaces, lc=2e-2, show_mesh=False):
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


def create_bspline_surface_mesh_file(surface, filepath, lc=2e-2, show_mesh=False):
    gmsh = create_bspline_surface_mesh(surface, lc=lc, show_mesh=show_mesh)
    gmsh.write(filepath)


def create_bspline_surface_mesh_model(surface, lc=2e-2, show_mesh=False):
    gmsh = create_bspline_surface_mesh(surface, lc=lc, show_mesh=show_mesh)
    return gmsh.model


def create_bspline_volume_mesh(surfaces, lc=2e-2, show_mesh=False):
    if not isinstance(surfaces, list):
        surfaces = [surfaces]

    gmsh.initialize()
    factory = gmsh.model.occ

    volume_tag, _ = create_holed_bspline_surface_volume(factory, surfaces, lc)
    gmsh.model.occ.heal_shapes()
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    if show_mesh and '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    return gmsh


def create_bspline_volume_mesh_file(surface, filepath, lc=2e-2, show_mesh=False):
    gmsh = create_bspline_volume_mesh(surface, lc=lc, show_mesh=show_mesh)
    gmsh.write(filepath)


def create_bspline_volume_mesh_model(surface, lc=2e-2, show_mesh=False):
    gmsh = create_bspline_volume_mesh(surface, lc=lc, show_mesh=show_mesh)
    return gmsh.model


def create_holed_bspline_curve_mesh(curves, lc=2e-2, show_mesh=False):
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


def create_holed_bspline_surface_volume(factory, surfaces, lc=2e-2):
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

    S1 = factory.add_plane_surface(start_curves)
    S2 = factory.add_plane_surface(end_curves)

    Sl_1 = factory.add_surface_loop([S1, surface_tags[0], S2])
    V1 = factory.add_volume([Sl_1])

    return [V1, point_tags]


def create_holed_bspline_volume_mesh(surfaces, lc=2e-2, show_mesh=False):
    if not isinstance(surfaces, list):
        surfaces = [surfaces]

    gmsh.initialize()
    factory = gmsh.model.occ

    volume_tag, _ = create_holed_bspline_surface_volume(factory, surfaces, lc)
    gmsh.model.occ.heal_shapes()
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    if show_mesh and '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    return gmsh