import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

import mlflow.sklearn
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import DataPreprocessor  # your existing code

def main():
    mlflow.set_experiment("student_performance_baseline")
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/student_habits_performance.csv")
    preproc = DataPreprocessor()
    X, y = preproc.preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": (RandomForestRegressor, {"n_estimators": 100, "max_depth": 8, "random_state": 42}),
        "GradientBoosting": (GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 8, "random_state": 42}),
        "ExtraTrees": (ExtraTreesRegressor, {"n_estimators": 100, "max_depth": 8, "random_state": 42})
    }

    for name, (Cls, params) in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_params(params)

            model = Cls(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse  = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2   = r2_score(y_test, preds)
            mlflow.log_metrics({"rmse": rmse, "r2": r2})

            # Save & log preprocessor
            preproc.save_preprocessor(f"models/preprocessor_{name}.joblib")
            mlflow.log_artifact(f"models/preprocessor_{name}.joblib", artifact_path="preprocessor")

            # Log the model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{name}] RMSE={rmse:.3f}, R2={r2:.3f}")

if __name__ == "__main__":
    main()
