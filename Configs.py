import numpy as np

def U2Net_get_configs(version = 1, n_channel = 3):
  """
    verion 1: normal with 44M params
    version2: lite with 1M params
    n_channel: number of input channels
  """
  if version == 1: 
    return {
      # cfgs for building RSUs and sides
      # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilation), side]}
      'stage1': ['En_1', (7, n_channel, 32, 64, 1), -1],
      'stage2': ['En_2', (6, 64, 32, 128, 1), -1],
      'stage3': ['En_3', (5, 128, 64, 256, 1), -1],
      'stage4': ['En_4', (4, 256, 128, 512, 1), -1],
      'stage5': ['En_5', (4, 512, 256, 512, 1), -1],
      'stage6': ['En_6', (4, 512, 256, 512, 1), 512],
      'stage5d': ['De_5', (4, 1024, 256, 512, 1), 512],
      'stage4d': ['De_4', (4, 1024, 128, 256, 1), 256],
      'stage3d': ['De_3', (5, 512, 64, 128, 1), 128],
      'stage2d': ['De_2', (6, 256, 32, 64, 1), 64],
      'stage1d': ['De_1', (7, 128, 16, 64, 1), 64],
    }
  elif version == 2:
    return {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, n_channel, 16, 64, 1), -1],
        'stage2': ['En_2', (6, 64, 16, 64, 1), -1],
        'stage3': ['En_3', (5, 64, 16, 64, 1), -1],
        'stage4': ['En_4', (4, 64, 16, 64, 1), -1],
        'stage5': ['En_5', (4, 64, 16, 64, 1), -1],
        'stage6': ['En_6', (4, 64, 16, 64, 1), 64],
        'stage5d': ['De_5', (4, 128, 16, 64, 1), 64],
        'stage4d': ['De_4', (4, 128, 16, 64, 1), 64],
        'stage3d': ['De_3', (5, 128, 16, 64, 1), 64],
        'stage2d': ['De_2', (6, 128, 16, 64, 1), 64],
        'stage1d': ['De_1', (7, 128, 16, 64, 1), 64],
    }

def model_get_args():
  return dict(
    input_size = 512,
    n_channel = 4,
    n_class = 4,
  )

def preprocess_get_bound():
  lows, highs = np.array([ 54., 133.,  48.,  56.]), np.array([ 564., 1120., 1512., 4259.])
  return lows, highs