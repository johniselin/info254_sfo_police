import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def exchange_coordinate(df, lon, lat, prefix):
    """
    return df with fixed coordinate
    """
    lon_fix_name = prefix + "_lon_fix"
    lat_fix_name = prefix + "_lat_fix"

    df[lon_fix_name] = df[lon]
    df[lat_fix_name] = df[lat]

    df.loc[df[lat] < 0, lon_fix_name] = df.loc[df[lat] < 0, lat]
    df.loc[df[lat] < 0, lat_fix_name] = df.loc[df[lat] < 0, lon]

    return df


def epsg_converter(geodf):
    """
    return geodataframe with crs
    """
    crs = {'init': 'epsg:4326'}
    geodf = geodf.to_crs(crs)

    return geodf


def df_2_geodf(df, crs, lon, lat):
    """
    return geodataframe
    """
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    geodf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    return geodf


def add_geometry(df, crs):
    """
    return dataframe with geometry
    """
    # load corresponding census geo data
    if "geoid10_tract" in df.columns:
        geodf_id = "geoid10"
        df_id = "geoid10_tract"
        path = "data/census2010_ sf_tracks.geojson"
    elif "geoid10_block" in df.columns:
        geodf_id = "geoid10"
        df_id = "geoid10_block"
        path = "data/census2010_ sf_blocks.geojson"

    # load corresponding census geo data
    cen_geodf = gpd.read_file(path)
    cen_geodf = cen_geodf[['geometry', geodf_id]]

    # add geometory
    df = pd.merge(df, cen_geodf, how="left", left_on=df_id, right_on=geodf_id)
    df.drop(geodf_id, axis=1, inplace=True)
    geodf = gpd.GeoDataFrame(df, crs=crs)

    return geodf


def add_weather(geodf, weather_day):
    """
    return dataframe with nearest weather station data
    """
    # add precipitation
    geodf["date"] = pd.to_datetime(geodf["datetime"].dt.date)
    geodf = pd.merge(geodf, weather_day, how="left", on="date")

    return geodf