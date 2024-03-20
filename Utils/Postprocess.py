import numpy as np

import tensorflow as tf
import shapely.geometry as geometry
from shapely import Polygon, buffer
from rasterstats import zonal_stats
import geopandas as gd
import rasterio as rs
import cv2
from skimage.morphology import skeletonize #, remove_small_holes, remove_small_objects
from .Window import predict_windows
import shapely as sl

   
def farm_predict_adapter(batch, model):
  ## dilated predict
  batch= batch[..., :3] # take the first 3 band
  pred = model.predict(batch, verbose = 0)
  pred = np.array(pred)
  # print(pred.shape)
  pred = pred[:, 0, ..., 2:3] # take prediction of farm class
  return pred

  return predict

def boundary_predict_adapter(batch, model):
  batch= batch[..., :3] # take the first 3 band
  pred = model.predict(batch, verbose = 0)
  pred = np.array(pred)
  pred = pred[0, :, ...]
  return pred


def to_binary_mask(path_in, path_out, threshold = 0.5, invert = False):
  """
    Take a threshold with value > threshold is mapped to 1., otherwise 0.
    if invert is True, the opposite is executed.
      Args:
        path_in: input path of tif file to read
        path_out: output path of binarized tif file to write to
        threshold: threshold for pixel value to map to 0 or 1
        invert: high value is mapped to 0 if invert is True
      Return:
        None
  """

  with rs.open(path_in) as src:
    out_meta = src.meta
    bin_mask = src.read()
    bin_mask = np.where(bin_mask > threshold, 1., 0.)
  with rs.open(path_out, "w", **out_meta) as dest:
    if invert:
        dest.write(1- bin_mask)
    else:
        dest.write(bin_mask)

def invert_mask(path_in, path_out):
    """
      invert a tif file: new_value = 1 - old_value
      Args:
        path_in: input tif file to read
        path_out: output tif file to write to
      Return:
        None
    """
    with rs.open(path_in) as src:
        out_meta = src.meta
        mask = 1 - src.read()
    with rs.open(path_out, "w", **out_meta) as dest:
        dest.write(mask)
## simplify polygon:
def filter_polygons(pathShape, pathSave, pathMask, farm_threshold = 0.4):

  """
  calculate the mean statistic of pixels value from a tif file (pathMask), that is contained within each polygon in a shape file (pathShape).
    Args:
      pathShape: path of shape file to read that contains polygons
      pathSave: path of shape file to write filtered polygons to
      pathMask: path of tif file that contains deciding value to filter polygons
      farm_threshold: threshold where polygons with a lower value will be removed
    Return:
      None
  """
  stats = zonal_stats(pathShape, pathMask,
            # stats="count min mean max median")
            stats="mean")
  
  gdf_filtered = gd.read_file(pathShape)

  remove_list = []
  for idx in range(len(stats)):
    if stats[idx]['mean'] is not None and stats[idx]["mean"] < farm_threshold:
      remove_list.append(idx)
  len(remove_list)
  
  gdf_filtered = gdf_filtered.drop(remove_list)
  gdf_filtered.to_file(pathSave)

## simplify polygon:
def simplify_polygons(path_in, path_out):
  """
    Simplify each polygon or multipolygon in a shape file (path_in) and write to another shape file (path_out)

      Args:
        path_in: shape file to read in
        path_out: shape file to write to
      Return:
        None
  
  """
  simplified_gdf = gd.read_file(path_in)

  for idx, row in simplified_gdf.iterrows():
    geom = row["geometry"]
    # area = geom.area
    envelop = sl.minimum_rotated_rectangle(geom)
    area = envelop.area
    length = envelop.length
    side_length = area/ length
    tolerance = min(4, side_length * 0.3)
    # tolerance = min(1., np.sqrt(area) /4)
    simple_geom = geom.simplify(tolerance, preserve_topology=False)
    # if simple_geom.area < eps:
    #   remove_list.append(idx)
    #   continue
    row["geometry"] = simple_geom

    simplified_gdf.loc[idx] = row

  simplified_gdf.to_file(path_out)



def get_angle(pt0, pt1, pt2):
    """
    Calculates the angle between three points.

    Args:
        pt1: The first point.
        pt2: The second point.
        pt0: The reference point.

    Returns:
        The angle in degrees.
    """

    # Calculate the vectors from the reference point to the other two points.
    v1 = np.array(pt1) - np.array(pt0)
    v2 = np.array(pt2) - np.array(pt0)

    # Calculate the angle between the vectors.
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # Convert the angle to degrees.
    angle = np.degrees(angle)

    # Return the angle.
    return angle


def checkAngle(concave_set, coords, threshold = 30):
  """
    Check all angle in a concave set (the continuous difference from a polygon with its convex hull), then return all the point that make a
    sharp angle with the start and end point at each side outside the concave set.
      Args:
        concave_set: continuous 1d array of point index in a polygon
        coords: array of coordinates of all the points in the polygon
        threshold: minimum angle to not be considered sharp
      Return:
        A array of index of point in the concave set that make a mentioned sharp angle
  """
  coords = np.array(coords)
  n = len(concave_set)
  start = concave_set[0]
  end = concave_set[n-1]
  # get 3 point to calculate angle
  if start == 0:
    idx1 = -1
  else:
    idx1 = start - 1
  if end == n - 1:
    idx2 = 0
  else:
    idx2 = end + 1

  sharp_indexs = []
  for idx0 in concave_set:

    pt0, pt1, pt2 = coords[idx0], coords[idx1], coords[idx2]
    angle1 = get_angle(pt0, pt1, pt2)
    pt0, pt1, pt2 = coords[idx0], coords[idx0-1], coords[(idx0+1)%n]
    angle2 = get_angle(pt0, pt1, pt2)
    # return angle

    if min(angle1, angle2) < threshold:
      sharp_indexs.append(idx0)

  return sharp_indexs


def refine_polygon(geom):
  # connect slender path (currently not implemented):
  #       get distance to extended cut
  #       get distance to mid point of cvhull dc
  #       if dedges > k * dc:
  #         extends
  #         create 2 new polygon
  #         recursion
  #       else:
  #         remove point
  #         recursion

  """
    Refine a polygon by removing all concave point that make a sharp angle
      Args:
        polygon of shapely
      Return:
        polygon of shapely with some mentioned point removed
  """

  # simple_geom = geom.simplify(4., preserve_topology=False)
  convexHull = geom.convex_hull
  geom_coords = list(geom.exterior.coords)

  convexHull_coords = list(convexHull.exterior.coords)
  concave_sets = []
  current_concave = False
  current_set = []
  for idx, point in enumerate(geom_coords):
      if point not in convexHull_coords:
        if current_concave:
          current_set.append(idx)
        else:
          current_concave = True
          current_set = [idx]
      else:
        if len(current_set) > 0:

          # plt.imshow()

          concave_sets.append(current_set)
          current_concave = False

  sharp_indexs = []
  for set_ in concave_sets:
    remove_indexs = checkAngle(set_, geom_coords)
    for index_ in remove_indexs:
      sharp_indexs.append(index_)


  keep_coords = []
  for idx, coord in enumerate(geom_coords):
    if idx not in sharp_indexs:
      keep_coords.append(coord)

  polygon = Polygon(keep_coords)

  return polygon

def refine_polygons(path_in, path_out = None, maximum_area = 100 * 1000.):

  """
    Refine all the polygon in a shape file with all the polygon that has too large area removed and refine the rest by removing 
    all the sharp angle.
      Args:
        path_in: input shape file contain the polygons or multipolygons to refine
        path_out: output shape file to write the new refined polygons or multipolygons to
      Return:
        None or the geopandas frame if path_out is None
  """
  gdf = gd.read_file(path_in)
  refined_gdf = gdf.copy()
  row_remove_list = []

  for idx, row in gdf.iterrows():

    # eps = 0.1
    geom = row["geometry"]
      
    if geom is None:
        row_remove_list.append(idx)
        continue

    if geom.geom_type == "MultiPolygon":
      # print(geom)
      geoms = list(geom.geoms)
      polygons = []
      for geom in geoms:

        if geom.area > maximum_area:
          row_remove_list.append(idx)
          break

        polygons.append(refine_polygon(geom))
      polygon = geometry.MultiPolygon(polygons)


    elif geom.geom_type == "Polygon":


      if geom.area > maximum_area:
        row_remove_list.append(idx)
        continue

      polygon = refine_polygon(geom)
    else:
        row_remove_list.append(idx)
        continue

    row["geometry"] = polygon

    refined_gdf.loc[idx] = row

  refined_gdf = refined_gdf.drop(row_remove_list)
  if path_out is None:
    return refined_gdf

  else:
    refined_gdf.to_file(path_out)

def refine_opening(path_in, path_out, size = 7):
    """
      refine a mask tif file with morphological opening
        Args:
          path_in: input tif file to read in
          path_out: output tif file to write to
        Return:
          None
    """
    with rs.open(path_in) as src:
        out_meta = src.meta
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel = np.ones((size,size)))
    with rs.open(path_out, "w", **out_meta) as dest:
        dest.write(image[np.newaxis, ...])
        
def refine_buffer(path_in, path_out, distance = 1.):
    """
      refine all the polygon or multipolygons in a shape file with buffering (shapely)
        Args:
          path_in: input shape file to read in
          path_out: output shape file to write to
        Return:
          None
    """
    gdf = gd.read_file(path_in)

    for idx, row in gdf.iterrows():
        geom = row["geometry"]
        if geom.geom_type == "MultiPolygon":
          # print(geom)
            geoms = list(geom.geoms)
            polygons = []
            for geom in geoms:
                geom = buffer(geom, distance = -distance)
                geom = buffer(geom, distance = distance)
                polygons.append(geom)
            polygon = geometry.MultiPolygon(polygons)

        else:
    
            polygon = buffer(geom, distance = -distance)
            polygon = buffer(polygon, distance= distance)

        row["geometry"] = polygon
        
        gdf.loc[idx] = row
    gdf.to_file(path_out)
    
    
def trim_paths(bin_mask, padding = 10, repeat = 5, length = 100):
  """
    Remove (change pixel to zeros) all the incomplete path in a binary of boundary (paths) mask using repeated skeletonize and removing endpoint
      Args:
        bin_mask: 2d binary mask
        padding: the amount of padding for the mask at the edge (to 1.) to avoid removal of incomplete path at the edge
        repeat: the amount of repeat each skeletonize and masking cycle
        length: the maximum expected length of paths to remove
      Return:
        new 2d binary mask with removed imcomplete paths
  """
  if len(bin_mask.shape) > 2:
    bin_mask = bin_mask[..., 0]
  kernel = np.ones((3,3)).astype(np.uint8)
  kernel5 = np.ones((5,5)).astype(np.uint8)
  for _ in range(repeat):
      # bin_mask = np.where(mask > threshold, 1., 0.)
      skeleton = skeletonize(bin_mask).astype(bool).astype(np.uint8)
      untrimmed = skeleton[padding:-padding, padding:-padding]
      untrimmed = np.pad(untrimmed, padding, mode='constant', constant_values=1)
      trimmed = untrimmed.astype(np.uint8)
      # plotN(untrimmed, trimmed, n_row = 1)
      for _ in range(length):
            trimmed = cv2.filter2D(trimmed.astype(np.uint8), -1, kernel = np.ones((3,3))) * trimmed
            trimmed = np.where(trimmed < 3, 0, 1)

      
      dif = np.where(untrimmed > trimmed, 1, 0)
      dil_dif = cv2.dilate(dif.astype(np.uint8), kernel, iterations = 1)
      bin_mask = np.where(dil_dif > 0, 0, bin_mask)
      bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel5)
      # plt.show()

# mask[padding:-padding, padding:-padding] = trimmed[padding:-padding, padding:-padding, np.newaxis]

  return bin_mask

    
def trim_paths_gpu(bin_mask, padding = 10, repeat = 5, length = 100):
  """
    Same as trim_paths but using gpu instead of cpu.
    Remove (change pixel to zeros) all the incomplete path in a binary of boundary (paths) mask using repeated skeletonize and removing endpoint
      Args:
        bin_mask: 2d binary mask
        padding: the amount of padding for the mask at the edge (to 1.) to avoid removal of incomplete path at the edge
        repeat: the amount of repeat each skeletonize and masking cycle
        length: the maximum expected length of paths to remove
      Return:
        new 2d binary mask with removed imcomplete paths
  """
  if len(bin_mask.shape) > 2:
    bin_mask = bin_mask[..., 0]
  kernel = np.ones((3,3)).astype(np.uint8)
  kernel5 = np.ones((5,5)).astype(np.uint8)
  kernel_gpu = cp.asarray(kernel)
  bin_mask_gpu = cp.asarray(bin_mask)
  for _ in range(repeat):
      # bin_mask = np.where(mask > threshold, 1., 0.)

      skeleton = skeletonize(bin_mask_gpu.get()).astype(bool).astype(np.uint8)

      untrimmed = skeleton[padding:-padding, padding:-padding]
      untrimmed = np.pad(untrimmed, padding, mode='constant', constant_values=1)
      trimmed = untrimmed.astype(np.uint8)

      untrimmed_gpu = cp.asarray(trimmed)
      trimmed_gpu=cp.asarray(trimmed)
      # plotN(untrimmed, trimmed, n_row = 1)
      for _ in range(length):
          # trimmed_gpu = convolve2d_gpu(trimmed_gpu, kernel_gpu)[1:-1, 1:-1] * trimmed_gpu
          # trimmed_gpu = cp.where(trimmed_gpu < 3, 0, 1)
          trimmed_gpu = cp.where(convolve2d_gpu(trimmed_gpu, kernel_gpu)[1:-1, 1:-1] < 3, 0, trimmed_gpu)
          # trimmed_gpu = (trimmed_gpu < 3, 0, 1)
      # print(trimmed.shape)
      # np.unique(skeleton_type)
      
      dif = cp.where(untrimmed_gpu > trimmed_gpu, 1, 0)
      dil_dif = grey_dilation(dif, size =3)
      # dil_dif = cv2.dilate(dif.astype(np.uint8), kernel, iterations = 1)
      bin_mask_gpu = cp.where(dil_dif > 0, 0, bin_mask_gpu)
      # bin_mask = cv2.morphologyEx(bin_mask_gpu.get(), cv2.MORPH_OPEN, kernel5)
      # bin_mask_gpu = grey_opening(bin_mask_gpu, structure = kernel)
      bin_mask_gpu = grey_opening(bin_mask_gpu, size = 5)
      # plt.show()

# mask[padding:-padding, padding:-padding] = trimmed[padding:-padding, padding:-padding, np.newaxis]

  return bin_mask_gpu.get()


def trim_paths_window(path_in, path_out, length = 100, repeat = 5, use_gpu = True):
  """
    Remove (change pixel to zeros) all the incomplete path in a binary (uint8) tif file of boundary (paths) mask using repeated skeletonize and removing endpoint,
    using trim_paths function
      Args:
        path_in: input tif file to read in
        path_out: output tif file to write to
        repeat: the amount of repeat each skeletonize and masking cycle
        length: the maximum expected length of paths to remove

      Return:
        None
  """
  if use_gpu:
    import cupy as cp
    from cupyx.scipy.ndimage import grey_opening, grey_dilation
    from cupyx.scipy.signal import convolve2d as convolve2d_gpu
    predictor = lambda batch: np.array([trim_paths_gpu(x, padding = 20, repeat = repeat, length= length) for x in batch])[..., np.newaxis]
  else:
    predictor = lambda batch: np.array([trim_paths(x, padding = 20, repeat = repeat, length = length) for x in batch])[..., np.newaxis]

  preprocess = lambda x: x
  
  predict_windows(pathTif = path_in, pathSave = path_out, predictor = predictor, preprocess = preprocess,
                window_size = 480, input_dim = 1, predict_dim = 1,
                output_type = "uint8", batch_size = 4)
