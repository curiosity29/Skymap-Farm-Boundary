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
    with rs.open(path_in) as src:
        out_meta = src.meta
        mask = 1 - src.read()
    with rs.open(path_out, "w", **out_meta) as dest:
        dest.write(mask)
## simplify polygon:
def filter_polygons(pathShape, pathSave, pathMask):

  stats = zonal_stats(pathShape, pathMask,
            # stats="count min mean max median")
            stats="mean")
  
  gdf_filtered = gd.read_file(pathShape)

  farm_threshold = 0.4
  remove_list = []
  for idx in range(len(stats)):
    if stats[idx]['mean'] is not None and stats[idx]["mean"] < farm_threshold:
      remove_list.append(idx)
  len(remove_list)
  
  gdf_filtered = gdf_filtered.drop(remove_list)
  gdf_filtered.to_file(pathSave)
  # return gdf_filtered

# gdf.drop(remove_list)
  
## simplify polygon:
def simplify_polygons(path_in, path_out):
  simplified_gdf = gd.read_file(path_in)
  # simplified_gdf = gdf.copy()
  # remove_list = []
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
    # print(row["geometry"])
    simplified_gdf.loc[idx] = row

  simplified_gdf.to_file(path_out)
  # return simplified_gdf


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
  """
  connect slender path (currently not implemented):
        get distance to extended cut
        get distance to mid point of cvhull dc
        if dedges > k * dc:
          extends
          create 2 new polygon
          recursion
        else:
          remove point
          recursion

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



  # print(len(sharp_indexs))
  # if len(keep_coords) != len(geom_coords):
  #   x, y = geom.exterior.xy
  #   plt.subplot(121)
  #   plt.plot(x, y)
  #   x, y = polygon.exterior.xy
  #   plt.subplot(122)
  #   plt.plot(x, y)
  #   plt.show()



  return polygon

def refine_polygons(path_in, path_out = None):
  """
    save path = None mean no saving and return result
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
        # if geom.area < eps:
        #     row_remove_list.append(idx)
        #     continue
        polygons.append(refine_polygon(geom))
      polygon = geometry.MultiPolygon(polygons)


    elif geom.geom_type == "Polygon":

      # if geom.area < eps:
      #   row_remove_list.append(idx)
      #   continue

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

def refine_closing(path_in, path_out, size = 7):
    with rs.open(path_in) as src:
        out_meta = src.meta
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel = np.ones((size,size)))
    with rs.open(path_out, "w", **out_meta) as dest:
        dest.write(image[np.newaxis, ...])
    
def refine_buffer(path_in, path_out, distance = 1.):

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
    
def trim_paths(mask, padding = 20, threshold = 0.5, repeat = 5):
    for _ in range(repeat):
        bin_mask = np.where(mask > threshold, 1., 0.)
        padding = 20
        skeleton = skeletonize(bin_mask).astype(bool).astype(np.uint8)
        untrimmed = skeleton[padding:-padding, padding:-padding, 0]
        untrimmed = np.pad(untrimmed, padding, mode='constant', constant_values=1)
        trimmed = untrimmed
        # plotN(untrimmed, trimmed, n_row = 1)
        for _ in range(100):
            trimmed = cv2.filter2D(trimmed.astype(np.uint8), -1, kernel = np.ones((3,3))) * trimmed
            trimmed = np.where(trimmed < 3, 0, 1)
        # print(trimmed.shape)
        # np.unique(skeleton_type)
        
        dif = np.where(untrimmed > trimmed, 1, 0)
        dil_dif = cv2.dilate(dif.astype(np.uint8), np.ones((3,3)), iterations = 1)
        mask = np.where(dil_dif > 0, 0, mask[..., 0])[..., np.newaxis]
        # plt.show()

  # mask[padding:-padding, padding:-padding] = trimmed[padding:-padding, padding:-padding, np.newaxis]

    return mask


def trim_paths_window(path_in, path_out, threshold = 0.5):
  predictor = lambda batch: np.array([trim_paths(x, padding = 20, threshold = threshold, repeat = 5) for x in batch])
  preprocess = lambda x: x
  
  predict_windows(pathTif = path_in, pathSave = path_out, predictor = predictor, preprocess = preprocess,
                window_size = 480, input_dim = 1, predict_dim = 1,
                output_type = "uint8", batch_size = 4)
