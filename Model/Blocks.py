import tensorflow as tf
from tensorflow.keras import layers

######################################
class CoBaRe(layers.Layer):
  """
    convolution, batch normalization, relu
  """
  def __init__(self, filters = 64, kernel_size = 3,
               padding = "same", dilation_rate = 1, use_bias = True, transpose = False,
               convArgs = {},
               **kwargs):
    super(CoBaRe, self).__init__(**kwargs)
    if transpose:
      self.conv = layers.Conv2DTranspose(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias, **convArgs)
    else:
      self.conv = layers.Conv2D(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias, **convArgs)
    self.batch = layers.BatchNormalization(axis = -1)
    # self.relu = layers.Activation('relu')

  def call(self, x):
    return tf.nn.relu(self.batch(self.conv(x)))

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "conv": self.conv,
        "batch": self.batch
      }
    )
    return config

  @classmethod
  def from_config(cls, config):
    config["conv"] = tf.keras.layers.deserialize(config["conv"])
    config["batch"] = tf.keras.layers.deserialize(config["batch"])
        
    return cls(**config)

######################################

class CoSigUp(layers.Layer):
  def __init__(self, up_size = 2, filters = 1, kernel_size = 3,
               padding = "same", dilation_rate = 1, use_bias = True, transpose = False,
               **kwargs):
    super(CoSigUp, self).__init__(**kwargs)
    if transpose:
      self.conv = layers.Conv2DTranspose(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias)
    else:
      self.conv = layers.Conv2D(
        filters = filters, kernel_size = kernel_size,
        padding = padding, dilation_rate = dilation_rate, use_bias = use_bias)
    self.upper = layers.UpSampling2D(up_size)
    # self.batch = layers.Up(axis = -1)
    # self.relu = layers.Activation('relu')

  def call(self, x):
    x = self.conv(x)
    x = tf.nn.sigmoid(x)
    return self.upper(x)

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "conv": self.conv,
        "upper": self.upper
      }
    )
    return config

  @classmethod
  def from_config(cls, config):
    config["conv"] = tf.keras.layers.deserialize(config["conv"])
    config["upper"] = tf.keras.layers.deserialize(config["upper"])      
    return cls(**config)
  
######################################