import rasterio as rs
from rasterio.features import dataset_features
import geopandas as gd
def vectorize(path_in, path_out):
    with rs.open(path_in) as src:
      
        gdf = gd.GeoDataFrame.from_features(dataset_features(src, bidx=1, as_mask=True, geographic = False))
        gdf.crs = src.meta["crs"]
        gdf.to_file(path_out)