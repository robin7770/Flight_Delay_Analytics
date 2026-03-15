import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


class ModelTrainer:

    def prepare_data(self, df, target):

        df_copy = df.copy()

        # Encode categorical columns
        for col in df_copy.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

        # Convert datetime to numeric
        for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
            df_copy[col] = df_copy[col].astype('int64') // 10**9

        # Handle missing values
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        df_copy = df_copy.fillna(0)

        X = df_copy.drop(columns=[target], errors='ignore')
        y = df_copy[target]

        return train_test_split(X, y, test_size=0.2, random_state=42)


    def evaluate(self, y_test, predictions):

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\nModel Performance")
        print("RMSE:", rmse)
        print("MAE :", mae)
        print("R2  :", r2)


    def train_random_forest(self, df, target):

        print("\nTraining Random Forest...")

        X_train, X_test, y_train, y_test = self.prepare_data(df, target)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        self.evaluate(y_test, predictions)

        return model


    def train_gradient_boosting(self, df, target):

        print("\nTraining Gradient Boosting...")

        X_train, X_test, y_train, y_test = self.prepare_data(df, target)

        model = GradientBoostingRegressor()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        self.evaluate(y_test, predictions)

        return model


    def train_xgboost(self, df, target):

        print("\nTraining XGBoost...")

        X_train, X_test, y_train, y_test = self.prepare_data(df, target)

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        self.evaluate(y_test, predictions)

        return model