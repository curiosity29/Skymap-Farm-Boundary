import geopandas as gdp
import os
import math
from shapely import geometry, wkt
def get_utm_from_wgs(lon, lat):
    """
    Use longitude, latitude of location for get EPSG code.

    Parameters
    ----------
    lon,lat :
        Longitude, latitude of location you want to get EPSG code

    Returns
    -------
    EPSG code of this location
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code1 = '326' + utm_band
    else:
        epsg_code1 = '327' + utm_band
    return epsg_code1
def create_list_shp(shapedir):
    list_shp = []
    for dir_name in os.listdir(shapedir):
        for file_name in os.listdir(os.path.join(shapedir,dir_name)):
            if file_name.endswith(".shp"):
                list_shp.append(os.path.join(shapedir,dir_name, file_name))
    return list_shp

def change_crs(input_path, bound_path = None, output_path = None):
    # input_path = r"/mnt/g/farm/Result_QC/MH_FandV/MH_FandV.shp"
    # output_path = r"/mnt/g/farm/Result_QC/MH_FandV/MH_FandV_utm.shp"
    # bound_path = r"/mnt/g/farm/MH_FandV.geojson"
    # print(output_path)
    crs = {'init':'epsg:4326'}
    data_shp = gdp.read_file(input_path)
    data_shp_4326 = data_shp.to_crs(crs)
    if bound_path is None:
      minx, miny, maxx, maxy = (data_shp_4326.total_bounds)
    else:
      bound_gdf = gdp.read_file(bound_path)

      bound_gdf_4326 = bound_gdf.to_crs(crs)

      data_shp_4326=gdp.overlay(data_shp_4326, bound_gdf_4326, how='intersection', keep_geom_type=None, make_valid=True)
    #   print(data_shp_4326)
      minx, miny, maxx, maxy = (data_shp_4326.total_bounds)

    bounds_pgon = geometry.box(minx, miny, maxx, maxy)
    centroid_point = bounds_pgon.centroid
    utm_code = get_utm_from_wgs(centroid_point.x, centroid_point.y)
    # print(utm_code)
    crs_2 = {'init':'epsg:{}'.format(utm_code)}
    # print(crs_2)
    data_shp_utm = data_shp_4326.to_crs(crs_2)
    # data_shp_utm["Area(Square meter)"] = data_shp_utm.area
    # data_shp_utm["Area(Acres)"] = data_shp_utm.area/4046.8564224
    # print(data_shp_utm)
    # data_shp_out = data_shp_utm[["Area(Square meter)","Area(Acres)","geometry"]]
    data_shp_out = data_shp_utm[["geometry"]]
    data_shp_out = gdp.GeoDataFrame(data_shp_out, geometry='geometry', crs=crs_2)
    # data_shp_out = data_shp_out.to_crs(crs)
    data_shp_out.to_file(output_path)

# caculator_area(input_path = pathSaveShape2, output_path = pathSaveShape2)