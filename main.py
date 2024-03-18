import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
import os, sys, glob

from Utils.Window import predict_windows
from Utils.Vectorize import vectorize
from Utils.Postprocess import farm_predict_adapter, boundary_predict_adapter, \
    filter_polygons, simplify_polygons, refine_polygons, to_binary_mask, trim_paths_window
from Model import U2Net
import Configs
import tensorflow as tf
import geopandas as gd

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path_boundary", type=str, default="./Checkpoints/*.h5", help="checkpoint file to create boundary mask")
    arg("--weight_path_farm", type=str, default="./Checkpoints/*.h5", help="checkpoint file to create farm mask")

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
            batch_size = 1, simplify_distance = 1., boundary_threshold = 0.5, farm_threshold = 0.5, sharp_angle = 30,
            search_path = True):

    if search_path:
        try:
            weight_path_boundary = glob.glob(f"{weight_path_boundary}", recursive=True)[0]
            weight_path_farm = glob.glob(f"{weight_path_farm}", recursive=True)[0]
    
            image_path = glob.glob(f"{image_path}", recursive=True)[0]
        except:
            print("image or weight not found")
            return


    ### get boundary mask
    preprocess = lambda x: x/255
    input_dim = 3
    predict_dim = 1

    boundary_model = tf.keras.models.load_model(weight_path_boundary)
    predictor = partial(boundary_predict_adapter, model = boundary_model)

    do_trim = True
    do_filter = False
    do_refine = True

    boundary_mask_path = os.path.join(save_path_folder, "raw_boundary.tif")
    predict_windows(pathTif = image_path, pathSave = boundary_mask_path, predictor = predictor, preprocess = preprocess,
                    window_size = 480, input_dim = input_dim, predict_dim = predict_dim,
                    output_type = "float32", batch_size = batch_size)


    ### get farm mask
    if do_filter:
        args = Configs.model_get_args()
        farm_model = U2Net(**args)
        farm_model.load_weights(weight_path_farm)
        predictor = partial(farm_predict_adapter, model = farm_model)

        farm_mask_path = os.path.join(save_path_folder, "raw_farm.tif")
        predict_windows(pathTif = image_path, pathSave = farm_mask_path, predictor = predictor, preprocess = preprocess,
                        window_size = 448, input_dim = input_dim, predict_dim = predict_dim,
                        output_type = "float32", batch_size = batch_size)
    
    last = boundary_binary_mask_path = os.path.join(save_path_folder, "binary_boundary.tif")
    to_binary_mask(path_in = boundary_mask_path, path_out=boundary_binary_mask_path, threshold=boundary_threshold)
    if do_trim:
        boundary_binary_mask_trimmed_path = os.path.join(save_path_folder, "binary_trimmed_boundary.tif")
        trim_paths_window(boundary_binary_mask_path, boundary_binary_mask_path)
        last = boundary_binary_mask_trimmed_path

    boundary_shape_path =  os.path.join(save_path_folder, "shape_boundary.shp")
    vectorize(path_in = last, path_out = boundary_shape_path)
    last = boundary_shape_path

    if do_filter:
        boundary_filtered_shape_path = os.path.join(save_path_folder, "shape_boundary_filtered.shp")
        gdf = filter_polygons(boundary_shape_path, boundary_filtered_shape_path, farm_mask_path)
        last = boundary_filtered_shape_path

    boundary_simplified_path = os.path.join(save_path_folder, "shape_boundary_simplified.shp")
    simplify_polygons(path_in = last, path_out=boundary_simplified_path)
    last = boundary_simplified_path

    if do_refine:
        refined_path = os.path.join(save_path_folder, "refined_prediction.shp")
        refine_polygons(gdf, save_path = refined_path)
        last = refined_path
    
    result_path = os.path.join(save_path_folder, "shape_boundary_results.shp")
    gdf = gd.read_file(last)
    gdf.to_file(result_path)

if __name__ == "__main__":
    main_args = get_main_args()
    predict(**vars(main_args))