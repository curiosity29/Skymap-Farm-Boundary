import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
import os, sys, glob

from Utils.Window import predict_windows
from Utils import Preprocess
from Utils.Rasterize import rasterize
from Utils.Postprocess import predict_adapter_boundary, predict_adapter_farm, filter_polygons, simplify_polygons, refine_polygons
from Model import U2Net
import Configs
import tensorflow as tf

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path_boundary", type=str, default="./Checkpoint_weights/*.h5", help="checkpoint file to create boundary mask")
    arg("--weight_path_farm", type=str, default="./Checkpoint_weights/*.h5", help="checkpoint file to create farm mask")

    arg("--image_path", type=str, default="./Images/*.tif", help="4 channel input tif file")
    arg("--save_path_folder", type=str, default="./Predictions/", help="folder to save prediction and other processing file")
    arg("--batch_size", type=int, default=4, help="batch size each predict, lowering to reduce memory requirement")
    arg("--boundary_threshold", type=float, default=0.5, help="threshold to create boundary mask from prediction")
    arg("--farm_threshold", type=float, default=0.5, help="threshold to filter out non-farm polygon from vectorized predicton")
    arg("--simplify_distance", type=float, default=1.0, help="tolerance to simplify farm boundary polygon")
    arg("--sharp_angle", type=float, default=30, help="minimum accepted angle (in degree) to exists inside polygon's convex hull, correspongding vertex is removed otherwise")
    return parser.parse_args()


def predict(image_path = "./image.tif", save_path_folder = "./Predictions", 
            weight_path_boundary = "./Checkpoint_weights/*.weights.h5", weight_path_farm = "./Checkpoint_weights/*.weights.h5",
            batch_size = 4, simplify_distance = 1., boundary_threshold = 0.5, farm_threshold = 0.5,
            search_path = True):

    # if search_path:
    #     try:
    #         weight_path = glob.glob(f"{weight_path}", recursive=True)[0]
    #         image_path = glob.glob(f"{image_path}", recursive=True)[0]
    #     except:
    #         print("image or weight not found")
    #         return

    ### get boundary mask
    preprocess = lambda x: x/255
    input_dim = 3
    predict_dim = 1

    boundary_model = tf.keras.models.load_model(weight_path_boundary)
    predictor = partial(predict_adapter_boundary, model = boundary_model)

    boundary_mask_path = os.path.join(save_path_folder, "raw_boundary.tif")
    predict_windows(pathTif = image_path, pathSave = boundary_mask_path, predictor = predictor, preprocess = preprocess,
                    window_size = 512, input_dim = input_dim, predict_dim = predict_dim,
                    output_type = "float32", batch_size = batch_size)


    ### get farm mask

    args = Configs.model_get_args()
    farm_model = U2Net(**args)
    farm_model.load_weights(weight_path_farm)
    predictor = partial(predict_adapter_farm, model = farm_model)

    farm_mask_path = os.path.join(save_path_folder, "raw_farm.tif")
    predict_windows(pathTif = image_path, pathSave = farm_mask_path, predictor = predictor, preprocess = preprocess,
                    window_size = 512, input_dim = input_dim, predict_dim = predict_dim,
                    output_type = "float32", batch_size = batch_size)

    boundary_shape_path =  os.path.join(save_path_folder, "shape_boundary.shp")
    rasterize(path_in = boundary_mask_path, path_out = boundary_shape_path)
    
    gdf = filter_polygons(boundary_shape_path, farm_mask_path)
    gdf = simplify_polygons(gdf)

    result_path = os.path.join(save_path_folder, "final_prediction.tif")
    refine_polygons(gdf, save_path = result_path)


if __name__ == "__main__":
    main_args = get_main_args()
    predict(**vars(main_args))