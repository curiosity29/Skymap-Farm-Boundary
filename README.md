# Skymap-Farm-Boundary
Farm boundary for 3 band satellite image with post processing refinement

## Input:
  rgb image in tif format
  
## Output:
  shape file that contains polygons for each farms in the image

## Usage:
Run the main python file with parameters or use all the default parameter with each file in its corresponding named folder:
```
python main.py
```
## Parameter:
weight_path_boundary: checkpoint file to create boundary mask
weight_path_farm: checkpoint file to create farm mask (currently not needed by default)

+ image_path: rgb input tif file
+ save_path_folder: folder to save prediction and other processing file
+ batch_size: batch size each predict, lowering to reduce memory requirement
+ boundary_threshold: threshold to create binary boundary mask from prediction
+ farm_threshold: threshold to filter out non-farm polygon from vectorized predicton
+ simplify_distance: tolerance to simplify farm boundary polygon (unused parameter, currently auto choosen dynamically for each polygon)
+ sharp_angle: minimum accepted angle (in degree) to exists inside polygon's convex hull, correspongding vertex is removed otherwise
+ use_cupy: using cupy to run gpu on the process of filtering out incomple paths or using normal cpu

Outline:

  + Predict boundary mask and farm mask (currently only use boundary mask)
  + Binarize the prediction mask by a threshold and use morphological opening on the mask
  + Remove incomplete paths of the mask using repeated unmasking with skeletonized mask
  + Vectorize the binary boundary mask into polygons
  + Remove polygons that has low farm probability according to farm mask (currently not used)
  + Simplify each polygon with a tolerance
  + Use buffering down then up of the same amount for better looking shape
  + Remove sharp concave angle by filtering out some point in each polygon
