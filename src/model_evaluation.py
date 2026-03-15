import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:

    def compare_models(self, results):

        df = pd.DataFrame(results)

        plt.figure(figsize=(8,5))
        sns.barplot(x="Model", y="RMSE", data=df)

        plt.title("Model Comparison (RMSE)")
        plt.savefig("plots/model_comparison.png")
        plt.show()


    def feature_importance(self, model, features):

        importances = model.feature_importances_

        df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=df)

        plt.title("Feature Importance")
        plt.savefig("plots/feature_importance.png")
        plt.show()


    def discover_patterns(self, df):

        plt.figure(figsize=(7,5))
        sns.scatterplot(
            x="baro_altitude",
            y="velocity",
            data=df
        )

        plt.title("Altitude vs Velocity Pattern")
        plt.savefig("plots/altitude_velocity.png")
        plt.show()