import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
import os, sys, glob

from Utils.Window import predict_windows
from Utils.Vectorize import vectorize
from Utils.Postprocess import farm_predict_adapter, boundary_predict_adapter, refine_buffer, refine_closing,\
    filter_polygons, simplify_polygons, refine_polygons, to_binary_mask, trim_paths_window, invert_mask
from Utils.changeCrs import change_crs
from Model import U2Net
import Configs
import tensorflow as tf
import geopandas as gd

def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path_boundary", type=str, default="./Checkpoints/Boundary/*.h5", help="checkpoint file to create boundary mask")
    arg("--weight_path_farm", type=str, default="./Checkpoints/Farm/*.h5", help="checkpoint file to create farm mask")

    arg("--image_path", type=str, default="./Images/*.tif", help="4 channel input tif file")
    arg("--save_path_folder", type=str, default="./Predictions/", help="folder to save prediction and other processing file")
    arg("--batch_size", type=int, default=1, help="batch size each predict, lowering to reduce memory requirement")
    arg("--boundary_threshold", type=float, default=0.4, help="threshold to create boundary mask from prediction")
    arg("--farm_threshold", type=float, default=0.4, help="threshold to filter out non-farm polygon from vectorized predicton")
    arg("--simplify_distance", type=float, default=1.0, help="tolerance to simplify farm boundary polygon")
    arg("--sharp_angle", type=float, default=30, help="minimum accepted angle (in degree) to exists inside polygon's convex hull, correspongding vertex is removed otherwise")
    return parser.parse_args()


def predict(image_path = "./image.tif", save_path_folder = "./Predictions", 
            weight_path_boundary = "./Checkpoint_weights/*.weights.h5", weight_path_farm = "./Checkpoint_weights/*.weights.h5",
            batch_size = 1, simplify_distance = 1., boundary_threshold = 0.5, farm_threshold = 0.5, sharp_angle = 30,
            search_path = True):

    do_trim = True
    do_filter = False
    do_refine = True
    do_change_crs = True
    do_buffer = True
    do_closing = True
    
    
    if search_path:
        try:
            weight_path_boundary = glob.glob(f"{weight_path_boundary}", recursive=True)[0]
            if do_filter:
                weight_path_farm = glob.glob(f"{weight_path_farm}", recursive=True)[0]
    
            image_path = glob.glob(f"{image_path}", recursive=True)[0]
        except Exception as e:
            print("image or weight not found")
            print(e)
            return


    ### get boundary mask
    preprocess = lambda x: x/255
    input_dim = 3
    predict_dim = 1

    boundary_model = tf.keras.models.load_model(weight_path_boundary)
    predictor = partial(boundary_predict_adapter, model = boundary_model)


    boundary_mask_path = os.path.join(save_path_folder, "raw_boundary.tif")
    if not os.path.exists(boundary_mask_path):
        predict_windows(pathTif = image_path, pathSave = boundary_mask_path, predictor = predictor, preprocess = preprocess,
                    window_size = 480, input_dim = input_dim, predict_dim = predict_dim,
                    output_type = "float32", batch_size = batch_size)

    
    ### get farm mask
    farm_mask_path = os.path.join(save_path_folder, "raw_farm.tif")
    if do_filter and not os.path.exists(farm_mask_path):
        args = Configs.model_get_args()
        farm_model = U2Net(**args)
        farm_model.load_weights(weight_path_farm)
        predictor = partial(farm_predict_adapter, model = farm_model)

        predict_windows(pathTif = image_path, pathSave = farm_mask_path, predictor = predictor, preprocess = preprocess,
                        window_size = 448, input_dim = input_dim, predict_dim = predict_dim,
                        output_type = "float32", batch_size = batch_size)
    
    boundary_binary_mask_path = os.path.join(save_path_folder, "binary_boundary.tif")
    if not os.path.exists(boundary_binary_mask_path):
        to_binary_mask(path_in = boundary_mask_path, path_out=boundary_binary_mask_path, threshold=boundary_threshold, invert = False)
    last = boundary_binary_mask_path
    if do_trim:
        boundary_binary_mask_trimmed_path = os.path.join(save_path_folder, "binary_trimmed_boundary.tif")
        if not os.path.exists(boundary_binary_mask_trimmed_path):
            trim_paths_window(boundary_binary_mask_path, boundary_binary_mask_trimmed_path, threshold = boundary_threshold)
        last = boundary_binary_mask_trimmed_path

        
    boundary_inverted_path =  os.path.join(save_path_folder, "inverted_boundary.tif")
    if not os.path.exists(boundary_inverted_path):
        invert_mask(path_in = last, path_out = boundary_inverted_path)
    last = boundary_inverted_path

    if do_closing:
        boundary_closing_path = os.path.join(save_path_folder, "boundary_closing.tif")
        if not os.path.exists(boundary_closing_path):
            refine_closing(path_in = last, path_out = boundary_closing_path)
        last = boundary_closing_path

    boundary_shape_path =  os.path.join(save_path_folder, "shape_boundary.shp")
    if not os.path.exists(boundary_shape_path):
        vectorize(path_in = last, path_out = boundary_shape_path)
    last = boundary_shape_path

    if do_change_crs:
        boundary_changed_crs = os.path.join(save_path_folder, "changed_crs_boundary.shp")
        change_crs(input_path = last, output_path = boundary_changed_crs)
        last = boundary_changed_crs
                   
    if do_filter:
        boundary_filtered_shape_path = os.path.join(save_path_folder, "shape_boundary_filtered.shp")
        if not os.path.exists(boundary_filtered_shape_path):
            filter_polygons(boundary_shape_path, boundary_filtered_shape_path, farm_mask_path)
        last = boundary_filtered_shape_path


    boundary_simplified_path = os.path.join(save_path_folder, "shape_boundary_simplified.shp")
    if not os.path.exists(boundary_simplified_path):
        simplify_polygons(path_in = last, path_out=boundary_simplified_path)
    last = boundary_simplified_path
    
    if do_buffer:
        buffered_boundary_path = os.path.join(save_path_folder, "shape_boundary_buffered.shp")
        if not os.path.exists(buffered_boundary_path):
            # print("\n buffering \n ")
            refine_buffer(path_in = last, path_out = buffered_boundary_path, distance = 3)
        last = buffered_boundary_path

    if do_refine:
        refined_path = os.path.join(save_path_folder, "shape_refined_boundary.shp")
        if not os.path.exists(refined_path):
            refine_polygons(path_in = last, path_out = refined_path)
        last = refined_path
    
    result_path = os.path.join(save_path_folder, "shape_boundary_results.shp")
    gdf = gd.read_file(last)
    gdf.to_file(result_path)

if __name__ == "__main__":
    main_args = get_main_args()
    predict(**vars(main_args))