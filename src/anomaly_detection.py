import pandas as pd
from sklearn.ensemble import IsolationForest


class FlightAnomalyDetector:

    def __init__(self):

        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )

    def detect(self, data):

        features = data[[
            "velocity",
            "baro_altitude",
            "vertical_rate"
        ]].fillna(0)

        self.model.fit(features)

        data["anomaly_score"] = self.model.decision_function(features)

        data["anomaly"] = self.model.predict(features)

        return data