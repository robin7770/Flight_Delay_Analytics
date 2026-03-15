import pandas as pd
import numpy as np
from src.feature_engineering import feature_engineering_pipeline

class FlightPreprocessor:
    """
    Efficient preprocessing pipeline for flight delay prediction
    - Handles missing data robustly
    - Encodes categorical features
    - Adds engineered features
    - Compatible with real-time streams
    """

    def __init__(self):
        self.cat_mappings = {}  # Save mapping for categorical encoding

    def clean_data(self, df):

    # Ensure expected columns exist
        required_columns = [
            'DepDelay','ArrDelay','Airline','Origin','Dest','ScheduledDepTime'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        df = df.dropna(subset=['ScheduledDepTime'], how='all')

        # Ensure delay columns exist
        if 'DepDelay' not in df.columns:
            df['DepDelay'] = 0

        if 'ArrDelay' not in df.columns:
            df['ArrDelay'] = 0

        # Convert to numeric safely
        df['DepDelay'] = pd.to_numeric(df['DepDelay'], errors='coerce').fillna(0)
        df['ArrDelay'] = pd.to_numeric(df['ArrDelay'], errors='coerce').fillna(0)

        # Clip negative delays
        df['DepDelay'] = df['DepDelay'].clip(lower=0)
        df['ArrDelay'] = df['ArrDelay'].clip(lower=0)

        return df

    def feature_engineering(self, df):

        if 'FlightDate' in df.columns:
            df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
            df['DayOfWeek'] = df['FlightDate'].dt.dayofweek
            df['Month'] = df['FlightDate'].dt.month
        else:
            df['DayOfWeek'] = 0
            df['Month'] = 0

        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        if 'Origin' in df.columns:
            congestion = df.groupby("Origin").size()
            df["AirportTraffic"] = df["Origin"].map(congestion)

        return df

    def encode_categorical(self, df):
        """Convert categorical columns to integer codes"""
        categorical_cols = ['Airline','Origin','Dest']
        for col in categorical_cols:
            if col not in self.cat_mappings:
                df[col] = df[col].astype('category')
                self.cat_mappings[col] = dict(enumerate(df[col].cat.categories))
                df[col] = df[col].cat.codes
            else:
                # Use saved mapping for consistency in live data
                mapping = {v:k for k,v in self.cat_mappings[col].items()}
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        return df

    def preprocess(self, df):
        """Full pipeline: clean -> feature engineer -> encode"""
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical(df)
        return df



