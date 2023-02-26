//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {1, 2};
//+
Line(6) = {1, 1};
//+
Coherence;
//+
Recursive Delete {
  Curve{1}; 
}
//+
Recursive Delete {
  Curve{1}; 
}
//+
Coherence;
//+
Coherence;
//+
Coherence;
//+
Point(6) = {0, 1, 0, 1.0};
//+
Point(7) = {1, 1, 0, 1.0};
//+
Point(8) = {1, 0, 0, 1.0};
//+
Line(9) = {5, 6};
//+
Line(10) = {6, 7};
//+
Line(11) = {7, 2};
//+
Line(12) = {7, 2};
//+
Line(13) = {7, 2};
//+
Line(14) = {7, 2};
//+
Line(15) = {7, 2};
//+
Line(16) = {7, 2};
//+
Line(17) = {7, 2};
//+
Line(18) = {2, 5};
//+
Line(19) = {2, 5};
//+
Line(20) = {5, 4};
//+
Line(21) = {5, 2};
//+
Line(22) = {2, 7};
//+
Line(23) = {2, 7};
//+
SetFactory("OpenCASCADE");
Rectangle(5) = {0.9, 1.4, 0.5, 1, 0.5, 0};
//+
SetFactory("OpenCASCADE");
//+
Circle(2) = {-12, 2.5, 0, 0.5, 0, 2*Pi};
