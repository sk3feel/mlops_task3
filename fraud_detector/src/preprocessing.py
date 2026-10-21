import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_train_data():
    logger.info('Loading training data...')

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    logger.info('Raw train data imported. Shape: %s', train.shape)

    # Add some features
    train = add_features(train)
    logger.info('Train data processed. Shape: %s', train.shape)

    return train


def haversine(lat1, lon1, lat2, lon2):
    rad = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return rad * c


def add_features(df):
    df_ = df.copy()
    df_['haversine'] = haversine(df_['lat'], df_['lon'],
                                 df_['merchant_lat'], df_['merchant_lon'])
    df_["transaction_time"] = pd.to_datetime(df_["transaction_time"])
    df_["year"] = df_["transaction_time"].dt.year
    df_["month"] = df_["transaction_time"].dt.month
    df_["day"] = df_["transaction_time"].dt.day
    df_["hour"] = df_["transaction_time"].dt.hour
    df_["minute"] = df_["transaction_time"].dt.minute
    df_["weekend"] = (df_["transaction_time"].dt.dayofweek > 4).astype(int)
    return df_.drop(columns=["transaction_time", "name_1", "name_2", "street"])


# Main preprocessing function
def run_preproc(input_df):
    # Add features
    output_df = add_features(input_df)

    trans_time = pd.to_datetime(input_df["transaction_time"])
    output_df["day_of_week"] = trans_time.dt.dayofweek
    accurate_hour = trans_time.dt.hour + (trans_time.dt.minute / 60)
    ang = 2 * np.pi * (accurate_hour / 24) 

    output_df["hour_sin"] = np.sin(ang)
    output_df["hour_cos"] = np.cos(ang)
    output_df["is_night"] = (trans_time.dt.hour < 7).astype(int)

    numeric_ls = ["amount", "population_city", "haversine"]
    for i in numeric_ls:
        values = pd.to_numeric(output_df[i], errors="raise")
        if (values < 0).any():
            raise ValueError(f"{i} содержит отриц значения")
        output_df[f"{i}_log"] = np.log1p(values)

    categorical_columns = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
    for column in categorical_columns:
        output_df[column] = (output_df[column].astype("string").str.strip()
                             .replace("", pd.NA).fillna("unknown").astype(str))

    # Return resulting dataset
    logger.info('Added new features. Output shape: %s', output_df.shape)
    return output_df
