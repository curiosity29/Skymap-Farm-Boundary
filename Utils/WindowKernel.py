import numpy as np
import rasterio as rs
# import tensorflow as tf
from rasterio.windows import Window as rWindow
from tqdm import tqdm
import scipy

class WindowExtractor():
  def __init__(self, image_shape, window_shape, step_divide = 1):
    self.image_shape = image_shape
    self.window_shape = window_shape
    self.index = 0
    self.step_divide = step_divide
    self.n_row = (image_shape[0] - window_shape[0])  // (window_shape[0] // step_divide) + 2 # + 2 at start and finish
    self.n_col = (image_shape[1] - window_shape[0]) // (window_shape[1] // step_divide) + 2 # + 2 at start and finish
    self.row = 0
    self.col = 0
    self.total = self.getTotal()

  def getTotal(self):
    return int(self.n_col * self.n_row)

  def getRowCol(self, index):
    return self.index // self.n_col, self.index % self.n_col

  def next(self):
    self.row, self.col = self.getRowCol(self.index)
    self.index += 1
    if self.index > self.total:
      return (None, None), (None, None)
    return self.getWindow(self.row, self.col)

  def toRowCol(self, corX, corY):
    """
      get row and col index from pixel coordinate this does account for the last image in each row
    """
    row = corX // (self.window_shape[0] // self.step_divide)
    col = corY // (self.window_shape[0] // self.step_divide)
    # corX = row * self.window_shape[0] // self.step_divide
    # corY = col * self.window_shape[0] // self.step_divide
    return row, col
  def getWindow(self, row, col):
    """
    return top left coordinate and corner type: None, (0, 0), (0,1), ...
    -1: not corner
    0: first (left or top)
    1: last (right or bottem)
    """
    corner_type = [-1, -1]
    # print("col: ", col, self.n_col)
    # if col == self.n_col:
    #   print("none")
    #   return (None, None), (None, None)

    # print(row, col)
    # corX, corY = 0, 0
    # posY, posX = self.index // self.n_col, self.index % self.n_col
    if row == self.n_row-1:
      corner_type[1] = 1
      corX = self.image_shape[0] - self.window_shape[0]
    else:
      corX = row * self.window_shape[0] // self.step_divide

    if col == self.n_col-1:
      corner_type[0] = 1
      corY = self.image_shape[1] - self.window_shape[1]
    else:
      corY = col * self.window_shape[1] // self.step_divide

    if row == 0:
      corner_type[0] = 0
    if col == 0:
      corner_type[1] = 0

    return (corX, corY), corner_type

# windowExtractor = WindowExtractor(image_shape = (5000, 5000), window_shape = (512, 512), step_divide = 1)
# for _ in range(110):
#   window = windowExtractor.next()
#   print(window)
#   # print(window[0])
#   if window[0][0] is None:
#     break

def create_kernel(kernel, image_W, image_H,  window_size = 512, count = 1, step_divide = 1.25, dtype = "uint16"):
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = step_divide)
    # create empty kernel
    with rs.open("./kernel.tif", "w", width = image_W, height = image_H, count = count, dtype = dtype) as dest:
        pass
        # while True:
        #     (corX, corY), corner_type = extractor.next()
        #     if corX is None:
        #         break
        #     window = rWindow(corX, corY, window_size, window_size)
        #     dest.write(np.zeros((1, window_size, window_size)), window = window)
    
    # create kernel weight
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = step_divide)
    
    with rs.open("./kernel.tif", "w+", width = image_W, height = image_H, count = count, dtype = dtype) as dest:
        
        while True:
            (corX, corY), corner_type = extractor.next()
            if corX is None:
                break
            window = rWindow(corX, corY, window_size, window_size)
            current = dest.read(window = window)[0]
            current += kernel.astype(dtype)
            dest.write(current[np.newaxis, ...], window = window)


def get_kernel(window_size):
    patch_weights = np.ones((window_size, window_size))
    patch_weights[0, :] = 0
    patch_weights[-1, :] = 0
    patch_weights[:, 0] = 0
    patch_weights[:, -1] = 0
    patch_weights = scipy.ndimage.distance_transform_edt(patch_weights) + 1
    # patch_weights = patch_weights[1:-1, 1:-1]
    kernel = patch_weights
    return kernel

def predict_windows_kernel(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, output_type = "int8", batch_size = 1, step_divide = 1.25, kernel = None):
    args = locals().copy()
    if kernel is None:
        kernel = get_kernel(window_size =window_size)

    args["kernel"] = kernel
    predict_windows(**args)
    
    with rs.open(pathSave) as src:
        meta = src.meta
        image_W = meta["width"]
        image_H = meta["height"]

    ## create big kernel 
    create_kernel(kernel, image_W, image_H,  window_size = window_size, count = 1, step_divide = step_divide, dtype = "uint16")
    
    # divide by kernel
                  
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = 1) # no need for overlapping
    with rs.open(pathSave, "r+") as dest:
        with rs.open("./kernel.tif", "r+") as big_kernel:
            
            while True:
                (corX, corY), corner_type = extractor.next()
                if corX is None:
                    break
                window = rWindow(corX, corY, window_size, window_size)
                kernel = big_kernel.read(window = window)
                pred = dest.read(window = window)
                pred = pred / kernel
                dest.write(pred, window = window)
                big_kernel.write(np.ones((1,window_size, window_size)), window = window)
            

def predict_windows(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, output_type = "int8", batch_size = 1, step_divide = 2, kernel = None):
  kernel = kernel[..., np.newaxis]
  with rs.open(pathTif) as src:
    # get meta
    out_meta = src.meta
    out_transform = src.transform
    profile = src.profile
    profile["transform"] = out_transform
    out_meta.update({"driver": "GTiff",
              "count": predict_dim, "dtype": output_type})

    image_W, image_H = out_meta["width"], out_meta["height"]
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = step_divide)
    with rs.open(pathSave, "w+", **out_meta) as dest:
      total = extractor.getTotal()
      total = total // batch_size if total % batch_size == 0 else total // batch_size + 1
      pogbar = tqdm(total = total) #, disable = True
      while True:
        batch = []
        windows = []
        for _ in range(batch_size):
          (corX, corY), corner_type = extractor.next()
          # print(corX, corY, corner_type)
          if corX is None:
            break
          window = rWindow(corX, corY, window_size, window_size)
          if corner_type == [-1, -1]:
            # windowWrite = rWindow(
            #   corX + window_size // 4, corY + window_size // 4, window_size//2, window_size//2)
            windows.append([window, True]) # is corner, write full size
          else:
            windows.append([window, False]) # not corner, write center

          image = src.read(window = window)
          image = np.transpose(image[:input_dim, ...], (1,2,0))
          image = preprocess(image)
          batch.append(image)
        if len(batch) == 0:
          break
        predicts = predictor(np.array(batch))

        for predict, window in zip(predicts, windows):
          predict = predict * kernel
          predict = np.transpose(predict, (2,0,1))
          # if window[1]:
            # predict = predict [
            #   :, window_size // 4 : - window_size // 4, window_size // 4: -window_size // 4]
          current = dest.read(window = window[0])
          predict = current + predict
          dest.write(predict, window = window[0])

        pogbar.update(1)
      pogbar.close()
