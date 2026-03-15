# src/feature_engineering.py

import pandas as pd
import numpy as np


def create_temporal_features(df):
    """
    Extract time-based features
    """

    df["Hour"] = df["ScheduledDepTime"]

    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek

    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    df["Month"] = df["FlightDate"].dt.month

    return df


def airport_congestion(df):
    """
    Calculate congestion at origin airport
    """

    congestion = df.groupby("Origin").size()

    df["AirportTraffic"] = df["Origin"].map(congestion)

    return df


def historical_delay_features(df):
    """
    Create statistical delay features
    """

    airline_delay = df.groupby("Airline")["DepDelay"].mean()

    df["AirlineAvgDelay"] = df["Airline"].map(airline_delay)

    route_delay = df.groupby(["Origin", "Dest"])["DepDelay"].mean()

    df["RouteKey"] = list(zip(df["Origin"], df["Dest"]))

    df["RouteAvgDelay"] = df["RouteKey"].map(route_delay)

    df.drop("RouteKey", axis=1, inplace=True)

    return df


def transform_distance(df):
    """
    Log transformation for skewed distance values
    """

    if "Distance" in df.columns:
        df["LogDistance"] = np.log1p(df["Distance"])

    return df


def feature_engineering_pipeline(df):
    """
    Complete feature engineering pipeline
    """

    df = create_temporal_features(df)

    df = airport_congestion(df)

    df = historical_delay_features(df)

    df = transform_distance(df)

    return df