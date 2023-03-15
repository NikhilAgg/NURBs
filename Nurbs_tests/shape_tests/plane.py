import gmsh
import sys
import numpy as np

gmsh.initialize()

# Set a target mesh size
lc = 2e-2

# Define the south B-Spline curve
P1  = gmsh.model.occ.addPoint(0.00, 0.00, 0.00, lc)
P2  = gmsh.model.occ.addPoint(1.00, 0.00, 0.00, lc)
P3  = gmsh.model.occ.addPoint(0.00, 1.00, 0.00, lc)
P4  = gmsh.model.occ.addPoint(1.00, 1.00, 0.00, lc)

points = [P1, P2, P3, P4]
# points = [P1, P2, P3, P4, P5, P6, P7, P8, P9]
weights = [1, 1, 1, 1]
knotsU = [0, 1]
knotsV = [0,1]
multiplicityU = [2, 2]
multiplicityV = [2, 2]

C1 = gmsh.model.occ.add_bspline_surface(points, 2, weights=weights, knotsU=knotsU, knotsV=knotsV, multiplicitiesU=multiplicityU, multiplicitiesV=multiplicityV)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)

# gmsh.write("t4.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()