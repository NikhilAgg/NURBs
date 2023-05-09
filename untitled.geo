//+
Circle(9) = {0, 0, 1, 5, 0, 2*Pi};
//+
Curve Loop(6) = {9};
//+
Curve Loop(7) = {9};
//+
Plane Surface(6) = {6, 7};
//+
BooleanFragments{ Curve{4}; Curve{5}; Delete; }{ }
//+
BooleanFragments{ Curve{2}; Curve{1}; Delete; }{ }
//+
BooleanFragments{ Point{2}; Curve{1}; Delete; }{ }
