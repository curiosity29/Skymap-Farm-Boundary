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

import cupy as cp
from cupyx.scipy.ndimage import grey_opening, grey_dilation
from cupyx.scipy.signal import convolve2d as convolve2d_gpu
    

def trim_paths_gpu(bin_mask, padding = 10, repeat = 5, length = 100, use_opening = False):
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
      if use_opening:
        bin_mask_gpu = grey_opening(bin_mask_gpu, size = 5)
      # plt.show()
# mask[padding:-padding, padding:-padding] = trimmed[padding:-padding, padding:-padding, np.newaxis]

  return bin_mask_gpu.get()

def trim_paths_window_gpu(path_in, path_out, length = 100, repeat = 5, use_gpu = True):
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
  predictor = lambda batch: np.array([trim_paths_gpu(x, padding = 20, repeat = repeat, length= length)[..., np.newaxis] for x in batch])

  preprocess = lambda x: x
  
  predict_windows(pathTif = path_in, pathSave = path_out, predictor = predictor, preprocess = preprocess,
                window_size = 480, input_dim = 1, predict_dim = 1,
                output_type = "uint8", batch_size = 4)
