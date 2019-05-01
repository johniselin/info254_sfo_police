import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import feather
from tqdm import tqdm


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


def load_rnn_data(path, window, predict_ts, geo_col=["geoid10_tract"], y_cols=["crime"]):
    """
    y_cols: ["crime"] or ["incident_type_0", "incident_type_1", "incident_type_2"]
    geo_col: ["geoid10_tract"] or ["geoid10_block"]
    return y_all and x_all of given path
    """
    # load data
    df = feather.read_dataframe(path)
    df.sort_values(by=["datetime", "geoid10_tract"], inplace=True)
    df.set_index("datetime", inplace=True)

    # input columns
    x_cols = list(df.drop(y_cols + geo_col, axis=1).columns)

    # group by geoid
    geo_grs = df.groupby(by=geo_col)

    # arrayes to store x and y
    # (window size, input size, no of tracts, no of timesteps)
    n_timesteps = int(len(df) / len(geo_grs)) - window - predict_ts + 1
    x_all = np.empty(shape=(window, len(x_cols + y_cols), len(geo_grs), n_timesteps))

    # (output size, no of tracts, no of timesteps)
    y_all = np.empty(shape=(len(y_cols), len(geo_grs), n_timesteps))

    # to store geo_ids and y_all's datetime
    geo_ids = []

    y_datetime = df.index.unique()[window + predict_ts - 1:]

    for i, (geo_id, gr) in enumerate(tqdm(geo_grs)):
        geo_ids.append(geo_id)
        x_values = gr[y_cols + x_cols].values
        y_values = gr[y_cols].values

        for j in range(window, len(gr) - predict_ts + 1):
            # generate x_all
            x_all[:, :, i, j - window] = x_values[j - window:j, :]
            y_all[:, i, j - window] = y_values[j + predict_ts - 1, :]

    return x_all, y_all, geo_ids, y_datetime