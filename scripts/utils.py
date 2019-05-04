import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import feather
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from copy import deepcopy
from datetime import datetime


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


def load_rnn_data(path, window, predict_ts, isdim3=True, geo_col=["geoid10_tract"], y_cols=["crime"]):
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
    # (no of timesteps, window size, no of tracts,  no of features, )
    n_timesteps = int(len(df) / len(geo_grs)) - window - predict_ts + 1
    x_all = np.empty(shape=(n_timesteps, window, len(geo_grs), len(x_cols + y_cols)))

    # (output size, no of tracts, no of outputs)
    y_all = np.empty(shape=(n_timesteps, len(geo_grs), len(y_cols)))

    # to store geo_ids and y_all's datetime
    geo_ids = []

    y_datetime = df.index.unique()[window + predict_ts - 1:]

    for i, (geo_id, gr) in enumerate(tqdm(geo_grs)):
        geo_ids.append(geo_id)
        x_values = gr[y_cols + x_cols].values
        y_values = gr[y_cols].values

        for j in range(window, len(gr) - predict_ts + 1):
            # generate x_all
            x_all[j - window, :, i, :] = x_values[j - window:j, :]
            y_all[j - window, i, :] = y_values[j + predict_ts - 1, :]

    if isdim3:
        x_all = np.reshape(x_all,
                           newshape=(x_all.shape[0], x_all.shape[1], x_all.shape[2] * x_all.shape[3]))
        y_all = np.reshape(y_all,
                           newshape=(y_all.shape[0], y_all.shape[1] * y_all.shape[2]))

    return x_all, y_all, geo_ids, y_datetime


def time_series_cv(x_all, y_all, n_splits=5, model=None, fit_params=None, baseline=False):
    """
    :param baseline: True or False (defualt: False)
    :return: train and test scores and prediction of y on test data
    """

    # prepare dictionary to store scores
    train_scores = {}
    metrics = ["acc", "log_loss"]
    for metric in metrics:
        train_scores[metric] = []
    test_scores = deepcopy(train_scores)

    # prepare dictionary to store predictions
    y_test_probs = np.zeros_like(y_all)

    # time series split
    tss = TimeSeriesSplit(n_splits=n_splits)

    for split, (train_idx, test_idx) in enumerate(tss.split(x_all, y_all)):

        print("---------- split {0} ----------".format(split))
        print("[{0:%H:%M:%S}] train_index:{1}~{2} test_index:{3}~{4}".format(
            datetime.now(), train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

        # create train and test set
        x_train = x_all[:train_idx[-1]]
        y_train = y_all[:train_idx[-1]]
        x_test = x_all[test_idx[0]:test_idx[-1]]
        y_test = y_all[test_idx[0]:test_idx[-1]]

        if baseline:
            # return 0 for all predicted probabiliby
            y_train_prob = np.zeros_like(y_train)
            y_test_prob = np.zeros_like(y_test)

            # return 0 for all binary predictions
            y_train_pred = np.zeros_like(y_train)
            y_test_pred = np.zeros_like(y_test)

        else:
            # train
            model.fit(x_train, y_train, **fit_params)

            # predict
            y_train_prob = model.predict(x_train)
            y_test_prob = model.predict(x_test)

            # convert form probability to binary
            y_train_pred = np.fix(y_train_prob)
            y_test_pred = np.fix(y_test_prob)

        # store test prediction
        y_test_probs[test_idx[0]:test_idx[-1]] = y_test_prob

        # calculate metrics
        train_log_loss = log_loss(y_train.flatten(), y_train_prob.flatten())
        test_log_loss = log_loss(y_test.flatten(), y_test_prob.flatten())
        train_acc = accuracy_score(y_train.flatten(), y_train_pred.flatten())
        test_acc = accuracy_score(y_test.flatten(), y_test_pred.flatten())

        # store scores
        train_scores["log_loss"].append(train_log_loss)
        test_scores["log_loss"].append(test_log_loss)
        train_scores["acc"].append(train_acc)
        test_scores["acc"].append(test_acc)

        print("[{0:%H:%M:%S}] train_log_loss:{1} test_log_loss:{2}".format(
            datetime.now(), train_log_loss, test_log_loss))
        print("[{0:%H:%M:%S}] train_acc:{1} test_acc:{2}\n".format(
            datetime.now(), train_acc, test_acc))

        # convert to dataframe
        train_scores_df = pd.DataFrame(train_scores)
        test_scores_df = pd.DataFrame(test_scores)

    return train_scores_df, test_scores_df, y_test_probs