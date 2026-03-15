import time
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest_live_data import fetch_live_flights
from src.preprocess import FlightPreprocessor
from src.feature_selection import FeatureSelector
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.logger import setup_logger


setup_logger()

preprocessor = FlightPreprocessor()


def run_pipeline():

    while True:

        print("Fetching live flights...")

        flights = fetch_live_flights()

        print("Columns from API:")
        print(flights.columns)

        if flights.empty:
            time.sleep(60)
            continue

        # -------------------------
        # DATA PREPROCESSING
        # -------------------------
        processed = preprocessor.preprocess(flights)

        # -------------------------
        # FEATURE SELECTION
        # -------------------------
        selector = FeatureSelector()

        selector.mutual_information(processed, target="velocity")
        selector.recursive_feature_elimination(processed, target="velocity")
        selector.tree_importance(processed, target="velocity")

        # Optional second target analysis
        selector.mutual_information(processed, target="DepDelay")
        selector.tree_importance(processed, target="DepDelay")

        # -------------------------
        # MODEL TRAINING
        # -------------------------
        trainer = ModelTrainer()

        model_rf = trainer.train_random_forest(processed, target="velocity")

        model_gb = trainer.train_gradient_boosting(processed, target="velocity")

        model_xgb = trainer.train_xgboost(processed, target="velocity")

        # -------------------------
        # MODEL EVALUATION
        # -------------------------
        evaluator = ModelEvaluator()

        evaluator.feature_importance(
            model_rf,
            processed.drop(columns=["velocity"]).columns
        )

        evaluator.discover_patterns(processed)

        # -------------------------
        # LIVE DATA SUMMARY
        # -------------------------
        print("\nLive Flight Summary")
        print("Flights processed:", len(processed))
        print("Unique airlines:", processed['Airline'].nunique())

        print("Top busy airports:")
        print(processed['Origin'].value_counts().head())

        # Run every 60 seconds
        time.sleep(60)


if __name__ == "__main__":
    run_pipeline()