import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class FeatureSelector:

    def prepare_data(self, df, target):

        df_copy = df.copy()

        # Encode categorical variables
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

        return X, y


    # -----------------------------
    # 1. FILTER METHOD
    # -----------------------------
    def mutual_information(self, df, target):

        X, y = self.prepare_data(df, target)

        mi_scores = mutual_info_regression(X, y)

        mi_df = pd.DataFrame({
            "Feature": X.columns,
            "MI Score": mi_scores
        })

        mi_df = mi_df.sort_values(by="MI Score", ascending=False)

        print("\nFilter Method (Mutual Information):")
        print(mi_df.head(10))

        return mi_df


    # -----------------------------
    # 2. WRAPPER METHOD
    # -----------------------------
    def recursive_feature_elimination(self, df, target):

        X, y = self.prepare_data(df, target)

        model = RandomForestRegressor(n_estimators=50, random_state=42)

        rfe = RFE(model, n_features_to_select=5)

        rfe.fit(X, y)

        selected_features = X.columns[rfe.support_]

        print("\nWrapper Method (RFE Selected Features):")
        print(selected_features)

        return selected_features


    # -----------------------------
    # 3. EMBEDDED METHOD
    # -----------------------------
    def tree_importance(self, df, target):

        X, y = self.prepare_data(df, target)

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(by="Importance", ascending=False)

        print("\nEmbedded Method (Tree Importance):")
        print(importance.head(10))

        return importance

        # -----------------------------
    # 4. CORRELATION FILTER
    # -----------------------------
    def correlation_filter(self, df, threshold=0.9):

        df_copy = df.copy()

        # Encode categorical variables
        for col in df_copy.select_dtypes(include='object').columns:
            df_copy[col] = df_copy[col].astype('category').cat.codes

        # Convert datetime to numeric
        for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
            df_copy[col] = df_copy[col].astype('int64') // 10**9

        df_copy = df_copy.fillna(0)

        corr_matrix = df_copy.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print("\nDropped correlated features:", to_drop)

        df_filtered = df.drop(columns=to_drop, errors="ignore")

        return df_filtered