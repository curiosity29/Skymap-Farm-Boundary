import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
import sys, glob

from Utils.Window import predict_windows
from Utils import Preprocess
from Utils.Postprocess import predict_adapter
from Model import MainModel
import Configs

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path", type=str, default="./Checkpoint_weights/*.h5", help="weights.h5 file")
    arg("--image_path", type=str, default="./Images/*.tif", help="4 channel input tif file")
    arg("--save_path", type=str, default="./Predictions/prediction.tif", help="1 channel output tif file")
    arg("--batch_size", type=int, default=4, help="batch size each predict, lowering to reduce memory requirement")
    arg("--strength", type=float, default=3.0, help="postprocessing parameter, amplify signal for small region")

    return parser.parse_args()


def predict(image_path = "./image.tif", save_path = "./prediction.tif", weight_path = "./checkpoint.weights.h5", 
            batch_size = 4, strength = 3, search_path = True):
    if search_path:
        try:
            weight_path = glob.glob(f"{weight_path}", recursive=True)[0]
            image_path = glob.glob(f"{image_path}", recursive=True)[0]
        except:
            print("image or weight not found")
            return

    lows, highs = Configs.preprocess_get_bound()
    preprocess = partial(Preprocess.preprocess, lows = lows, highs = highs)
    input_dim = 4
    predict_dim = 1
    args = Configs.model_get_args()
    
    def get_model(weight_path, args):
        model = MainModel.U2Net_dilated(**args)
        model.load_weights(weight_path)
        return model

    model = get_model(weight_path, args)
    predictor = partial(predict_adapter, model = model, strength = strength)

    predict_windows(pathTif = image_path, pathSave = save_path, predictor = predictor, preprocess = preprocess,
                    window_size = 512, input_dim = input_dim, predict_dim = predict_dim,
                    output_type = "int8", batch_size = batch_size)

if __name__ == "__main__":
    main_args = get_main_args()
    predict(**vars(main_args))