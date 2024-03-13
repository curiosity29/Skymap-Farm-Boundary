import numpy as np
import rasterio as rs
# import tensorflow as tf
from rasterio.windows import Window as rWindow
from tqdm import tqdm

class WindowExtractor():
  def __init__(self, image_shape, window_shape, step_divide = 1):
    self.image_shape = image_shape
    self.window_shape = window_shape
    self.index = 0
    self.step_divide = step_divide
    self.n_col = (image_shape[0] // window_shape[0] - 1) * step_divide + 2 # + 2 at start and finish
    self.n_row = (image_shape[1] // window_shape[1] - 1) * step_divide + 2 # + 2 at start and finish

  def getTotal(self):
    return int(self.n_col * self.n_row)

  def getWindow(self):
    """
    return top left coordinate and corner type: None, (0, 0), (0,1), ...
    """
    corner_type = [-1, -1]
    if self.index >= (self.n_col) * self.n_row:
      return (None, None), (None, None)

    corX, corY = 0, 0
    posY, posX = self.index // self.n_col, self.index % self.n_col
    if posX == self.n_col-1:
      corner_type[1] = 1
      corX = self.image_shape[0] - self.window_shape[0]
    else:
      corX = posX * self.window_shape[0] // self.step_divide
    if posY == self.n_row-1:
      corner_type[0] = 1
      corY = self.image_shape[1] - self.window_shape[1]
    else:
      corY = posY * self.window_shape[0] // self.step_divide

    if posY == 0:
      corner_type[1] = 0
    if posX == 0:
      corner_type[0] = 0

    self.index += 1
    return (corX, corY), corner_type


def predict_windows(pathTif, pathSave, predictor, preprocess, window_size = 512, input_dim = 3, predict_dim = 1, output_type = "float32", batch_size = 4):

  with rs.open(pathTif) as src:
    # get meta
    out_meta = src.meta
    out_transform = src.transform
    profile = src.profile
    profile["transform"] = out_transform
    out_meta.update({"driver": "GTiff",
              "count": predict_dim, "dtype": output_type})

    ##### ONLY TESTED FOR step_divide = 2 or 1
    step_divide = 2
    image_W, image_H = out_meta["width"], out_meta["height"]
    extractor = WindowExtractor(image_shape=(image_W, image_H), window_shape = (window_size, window_size), step_divide = 2)
    with rs.open(pathSave, "w", **out_meta) as dest:
      total = extractor.getTotal()
      total = total // batch_size if total % batch_size == 0 else total // batch_size + 1
      pogbar = tqdm(total = total) #, disable = True
      while True:
        batch = []
        windows = []
        for _ in range(batch_size):
          (corX, corY), corner_type = extractor.getWindow()
          # print(corX, corY, corner_type)
          if corX is None:
            break
          window = rWindow(corX, corY, window_size, window_size)
          if corner_type == [-1, -1]:
            windowWrite = rWindow(
              corX + window_size // 4, corY + window_size // 4, window_size//2, window_size//2)
            windows.append([windowWrite, True]) # is corner, write full size
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
          predict = np.transpose(predict, (2,0,1))
          if window[1]:
            predict = predict[
              :, window_size // 4 : - window_size // 4, window_size // 4: -window_size // 4]

          dest.write(predict, window = window[0])

        pogbar.update(1)
      pogbar.close()

