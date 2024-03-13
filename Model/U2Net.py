import tensorflow as tf
from tensorflow.keras import layers
from .Blocks import CoBaRe, CoSigUp
import Configs

##################

class RSU(layers.Layer):
  def __init__(self, height, in_ch, mid_ch, out_ch, dilations = None, pooling = False, **kwargs):

    super(RSU, self).__init__(**kwargs)
    if dilations is None:
      dilations = [1] * height + [2] + [1] * (height - 1)

    self.args = (height, in_ch, mid_ch, out_ch, dilations.copy(), pooling)
    self.height = height
    self.dilations = dilations
    self.pooling = pooling
    self.rs_dict = self.get_rs_dict


  def build(self, input_shape, loading = False, **kwargs):
    super().build(input_shape, **kwargs)
    self.model = self.get_model(*self.args, loading = loading)

  def build_from_config(self, config):
    super().build_from_config(config)
    # for 
    self.build(self, loading = True)

    # return build_config

  def get_rs_dict(self):
    rs_dict = {}
    for idx in range(1, self.height + 1):
      layer_name = f"rs_en_{idx}"
      rs_dict[layer_name] = getattr(self, layer_name)

    for idx in range(self.height, 0, -1):
      layer_name = f"rs_de_{idx}"
      rs_dict[layer_name] = getattr(self, layer_name)
    return rs_dict

  # staticmethod
  def get_model(self, height, in_ch, mid_ch, out_ch, dilations, pooling, loading = False):
    def cbr(x, name, filters = mid_ch, **args):
      dilation_rate = dilations.pop()
      if loading:
        layer = getattr(self, name)
      else:
        layer = CoBaRe(filters = filters, dilation_rate= dilation_rate,**args)
        setattr(self, name, layer)
      return layer(x)


    input = tf.keras.Input(shape = (None, None, in_ch), name = "RSU_input")
    x = cbr(input, filters = out_ch, name = "rs_en_1")
    skips = [x]
    x = cbr(x, name = "rs_en_2")
    skips.append(x)
    ## down
    for idx in range(3, height+1):
      if pooling:
        x = layers.MaxPool2D(2, name = f"down_{idx}")(x)
      x = cbr(x, name = f"rs_en_{idx}")
      skips.append(x)
      # print(x.shape)

    # middle
    x = cbr(x, name = f"rs_en_{height}")

    x = layers.Concatenate(axis = -1, name = f"skip_{height-1}")([x, skips.pop()])
    x = cbr(x, name = f"rs_de_{height-1}")
    # print("decoding")
    for idx in range(height-2, 1, -1):
      if pooling:
        x = layers.UpSampling2D(2, name = f"up_{idx}")(x)
      # print(x.shape)
      x = layers.Concatenate(axis = -1, name = f"skip_{idx}")([x, skips.pop()])
      x = cbr(x, name = f"rs_de_{idx}")
    if pooling:
      x = layers.UpSampling2D(2, name = f"up_1")(x)
    x = layers.Concatenate(axis = -1, name = "skip_1")([x, skips.pop()])
    x = cbr(x, name = "rs_de_1", filters = out_ch)
    output = layers.Add(name = "add_ouput")([x, skips.pop()])

    return tf.keras.Model(input, output)

  def get_config(self):
    config = super().get_config()
    # self.get_rs_dict()
    config.update(self.rs_dict)

    return config

  def call(self, inputs):
    return self.model(inputs)

###############
  
def get_side(x, filters = 2, kernel_size = 3, target_size = 256, **args):
  return CoSigUp(up_size = target_size // x.shape[1], kernel_size = 3, filters = filters, **args)(x)


def U2Net_augment(inputs, n_channel = 3, n_class = 2, input_size = 256, output_size = 256,
  softmax_head = True, version = 1, coarse_map = None, check_ignore = False, **ignore):
  ## sanity checking
  if check_ignore:
    if(len(ignore.keys()) == 0):
      print("nothing ignored")
    else:
      print("ignored:")
      print(list(ignore.keys()))

  ## get config from version 
  configs = Configs.U2Net_get_configs(version, n_channel=n_channel)
  n_en = (len(configs) - 3) // 2
  configList = list(configs.values())
  configs_en, configs_middle, configs_de = \
  configList[:n_en], configList[n_en: n_en + 3], configList[-n_en:]

  ## input and external model
  # inputs = tf.keras.Input(shape = (input_size, input_size, n_channel))

  cmap_input = coarse_map
  cmap = cmap_input

  # fix in_channel size
  in_ch_aug = [cmap.shape[-1]] * 11 
  in_ch_aug = list(reversed(in_ch_aug)) # reverse to use by pop

  x = layers.Concatenate(axis = -1, name = "aug_input")([inputs, cmap])

  skips = []
  sides = []
  for idx, config_ in enumerate(configs_en):
    name, (height, in_ch, mid_ch, out_ch, dilation), side = config_
    in_ch += in_ch_aug.pop()
    # print(f"shape: {x.shape}, in_ch: {in_ch}, out_ch: {out_ch}")
    x = RSU(height, in_ch, mid_ch, out_ch, dilations = None, name = name)(x)
    x = layers.MaxPool2D(2, name = f"down_{idx+1}")(x)
    cmap = layers.MaxPool2D(2, name = f"cmap_down_{idx+1}")(cmap)
    x = layers.Concatenate(axis = -1, name = f"aug_{idx+1}")([x, cmap])
    skips.append(x) # skip along with cmap
  # print(f"shape: {x.shape}, in_ch: {in_ch}, out_ch: {out_ch}")
  ### bottle neck
  # print("mid")
  middle_dilations = [1,1,2,4,8,4,2,1]
  for config_ in configs_middle:
    name, (height, in_ch, mid_ch, out_ch, dilation), side = config_
    # print(f"shape: {x.shape}, in_ch: {in_ch}, out_ch: {out_ch}")
    if name == "En_5":

      x = RSU(height, in_ch = in_ch + in_ch_aug.pop(), mid_ch = mid_ch, out_ch = out_ch, name = name, dilations = middle_dilations)(x)
      x = layers.Concatenate(axis = -1, name = f"aug_5")([x, cmap])
      skips.append(x)
      x = layers.MaxPool2D(2, name = "down_5")(x)
      # cmap = layers.MaxPool2D(2, name = f"cmap_down_5")(cmap)

    elif name == "En_6":
      x = RSU(height, in_ch = in_ch + in_ch_aug.pop(), mid_ch = mid_ch, out_ch = out_ch, name = name, dilations = middle_dilations)(x)
      # print(f"mid side: {get_side(x, filters = n_class, target_size = output_size).shape}")
      sides.append(get_side(x, filters = n_class, target_size = output_size, name = "side_6"))
      x = layers.UpSampling2D(2, name = "up_6")(x)
    elif name == "De_5":
      x = layers.Concatenate(axis = -1, name = "skip_5")([x, skips.pop()])
      x = RSU(height, in_ch = in_ch + in_ch_aug.pop(), mid_ch = mid_ch, out_ch = out_ch, name = name, dilations = middle_dilations)(x)
      
      # x = layers.UpSampling2D(2)(x)
      sides.append(get_side(x, filters = n_class, target_size = output_size, name = "side_5"))
  # print("decode")
  for idx, config_ in enumerate(configs_de):
    name, (height, in_ch, mid_ch, out_ch, dilation), side = config_
    in_ch += in_ch_aug.pop()
    # print(f"shape: {x.shape}, in_ch: {in_ch}, out_ch: {out_ch}")
    x = layers.Concatenate(axis = -1, name = f"skip_{n_en - idx}")([x, skips.pop()])
    x = layers.UpSampling2D(2, name = f"up_{n_en - idx}")(x)
    x = RSU(height, in_ch, mid_ch, out_ch, dilations = None, name = name)(x)
    sides.append(get_side(x, filters = n_class, target_size = output_size, name = f"side_{n_en - idx}"))
  # for side_ in sides:
  #   print(side_.shape)

  fused_maps = layers.Concatenate(axis = -1, name = "fused_map")(sides)
  fused_maps = layers.Concatenate(axis = -1, name = "aug_fused_map")([fused_maps, cmap_input])

  # print(f"maps: {maps.shape}")
  fused_output = layers.Conv2D(filters = n_class, kernel_size = 1, name = "fused_output")(fused_maps)
  fused_output = layers.Activation("sigmoid", name = "sigmoid_head")(fused_output)
  stacked_output = tf.stack([fused_output] + sides, axis = 1)
  # sides_output = list(map(tf.nn.sigmoid, sides))

  return stacked_output
