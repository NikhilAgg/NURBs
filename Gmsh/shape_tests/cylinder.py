import gmsh
import sys
import numpy as np

gmsh.initialize()

# Set a target mesh size
lc = 2e-2

# Define the south B-Spline curve
P1  = gmsh.model.occ.addPoint(1.00, 0.00, 0.00, lc)
P2  = gmsh.model.occ.addPoint(1.00, 1.00, 0.00, lc)
P3  = gmsh.model.occ.addPoint(0.00, 1.00, 0.00, lc)
P4  = gmsh.model.occ.addPoint(-1.00, 1.00, 0.00, lc)
P5  = gmsh.model.occ.addPoint(-1.00, 0.00, 0.00, lc)
P6  = gmsh.model.occ.addPoint(-1.00, -1.00, 0.00, lc)
P7  = gmsh.model.occ.addPoint(0.00, -1.00, 0.00, lc)
P8  = gmsh.model.occ.addPoint(1.00, -1.00, 0.00, lc)
P9  = gmsh.model.occ.addPoint(0.999, -0.001, 0.00, lc)

# Define the south B-Spline curve
P10  = gmsh.model.occ.addPoint(1.00, 0.00, 1.00, lc)
P11  = gmsh.model.occ.addPoint(1.00, 1.00, 1.00, lc)
P12  = gmsh.model.occ.addPoint(0.00, 1.00, 1.00, lc)
P13  = gmsh.model.occ.addPoint(-1.00, 1.00, 1.00, lc)
P14  = gmsh.model.occ.addPoint(-1.00, 0.00, 1.00, lc)
P15  = gmsh.model.occ.addPoint(-1.00, -1.00, 1.00, lc)
P16  = gmsh.model.occ.addPoint(0.00, -1.00, 1.00, lc)
P17  = gmsh.model.occ.addPoint(1.00, -1.00, 1.00, lc)
P18  = gmsh.model.occ.addPoint(0.999, -0.001, 1.00, lc)

points = [P1, P2, P3, P4, P5, P6, P7, P8, P1, P10, P11, P12, P13, P14, P15, P16, P17, P10]
# points = [P1, P2, P3, P4, P5, P6, P7, P8, P9]
weights = [1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1, 1/2**0.5, 1]
knotsU = [0, 1/4, 1/2, 3/4, 1]
knotsV = [0, 1]
multiplicityU = [2, 2, 2, 2, 2]
multiplicityV = [2, 2]

C1 = gmsh.model.occ.add_bspline_surface(points, 9, degreeU=2, degreeV=1, weights=weights, knotsU=knotsU, knotsV=knotsV, multiplicitiesU=multiplicityU, multiplicitiesV=multiplicityV)
# C2 = gmsh.model.occ.add_line(P1, P9)
# W1 = gmsh.model.occ.addCurveLoop([C1, -C2])
# W1 = gmsh.model.occ.addCurveLoop([C1])
# gmsh.model.occ.add_plane_surface([W1], 1)


gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

# gmsh.write("t4.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()