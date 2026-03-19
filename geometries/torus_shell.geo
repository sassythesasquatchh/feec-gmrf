SetFactory("OpenCASCADE");

R  = 1.0;   // major radius
r  = 0.30;  // minor radius
lc = 0.05;  // target mesh size

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

Torus(1) = {0, 0, 0, R, r, 2*Pi};
s[] = Boundary{ Volume{1}; };
Delete{ Volume{1}; };