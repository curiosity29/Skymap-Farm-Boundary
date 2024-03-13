# Skymap-Farm-Boundary
Farm boundary for 3 band satellite image with post processing refinement

Outline:

+ Predict boundary mask and farm mask
+ rasterize boundary mask into polygons
+ remove polygons that has low farm probability according to farm mask
+ simplify each polygon with a tolerance
+ remove sharp concave area in each polygon
