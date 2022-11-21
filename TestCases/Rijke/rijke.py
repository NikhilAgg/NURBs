import gmsh
import sys

def create_cylinder():
    gmsh.initialize()

    lc=1e-1
    gmsh.model.geo.add_point(0, 0, 0.5, lc, 1)
    gmsh.model.geo.add_point(0, 0.5, 0.5, lc, 2)
    gmsh.model.geo.add_point(0, 1, 0.5, lc, 3)

    gmsh.model.geo.add_point(1, 0, 0.5, lc, 4)
    gmsh.model.geo.add_point(1, 0.5, 0.5, lc, 5)
    gmsh.model.geo.add_point(1, 1, 0.5, lc, 6)

    gmsh.model.geo.add_circle_arc(1, 2, 3, 1, 1, 0, 0)
    gmsh.model.geo.add_circle_arc(3, 2, 1, 2, 1, 0, 0)

    gmsh.model.geo.add_circle_arc(4, 5, 6, 3, 1, 0, 0)
    gmsh.model.geo.add_circle_arc(6, 5, 4, 4, 1, 0, 0)

    gmsh.model.geo.add_curve_loop([1, 2], 1)
    gmsh.model.geo.add_curve_loop([3, 4], 2)
    gmsh.model.geo.add_plane_surface([1], 1)
    gmsh.model.geo.add_plane_surface([2], 2)

    gmsh.model.geo.add_line(1, 4, 5)
    gmsh.model.geo.add_line(3, 6, 6)

    gmsh.model.geo.add_curve_loop([1, 6, -3, -5], 3)
    gmsh.model.geo.add_curve_loop([2, 5, -4, -6], 4)

    gmsh.model.geo.add_surface_filling([3], 3)
    gmsh.model.geo.add_surface_filling([4], 4)

    gmsh.model.geo.add_surface_loop([1, 3, 2, 4], 1)
    gmsh.model.geo.add_volume([1], 1)
    # ov = gmsh.model.geo.extrude([(2, 5)], 1, 0, 0, [10], [1])
    gmsh.model.geo.synchronize()


    gmsh.model.add_physical_group(2, [1], 1)
    gmsh.model.add_physical_group(2, [2], 2)
    gmsh.model.add_physical_group(2, [3, 4], 3)

    gmsh.model.mesh.generate(3)
    return gmsh.model