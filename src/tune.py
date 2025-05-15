import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess import DataPreprocessor

# 1) Prepare data
df = pd.read_csv("data/student_habits_performance.csv")
preproc = DataPreprocessor()
X, y = preproc.preprocess_data(df, is_training=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) Define per-model search spaces
search_spaces = {
    "RandomForest": {
        "n_estimators": hp.choice("rf_n", [50, 100, 200]),
        "max_depth":    hp.quniform("rf_d", 5, 20, 1),
        "random_state": 42
    },
    "GradientBoosting": {
        "n_estimators": hp.choice("gb_n", [50, 100, 200]),
        "max_depth":    hp.quniform("gb_d", 5, 20, 1),
        "random_state": 42
    },
    "ExtraTrees": {
        "n_estimators": hp.choice("et_n", [50, 100, 200]),
        "max_depth":    hp.quniform("et_d", 5, 20, 1),
        "random_state": 42
    }
}

mlflow.set_experiment("student_performance_tuning")
best_overall = {"model": None, "rmse": float("inf"), "params": None}

# 3) Tune each model for 30 trials
for model_name, space in search_spaces.items():
    def objective(params):
        params["max_depth"] = int(params["max_depth"])
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)

            if model_name == "RandomForest":
                m = RandomForestRegressor(**params)
            elif model_name == "GradientBoosting":
                m = GradientBoostingRegressor(**params)
            else:
                m = ExtraTreesRegressor(**params)

            m.fit(X_train, y_train)
            preds = m.predict(X_val)
            rmse  = np.sqrt(mean_squared_error(y_val, preds))
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(m, artifact_path="model")
            return {"loss": rmse, "status": STATUS_OK}

    trials = Trials()
    best   = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)
    losses = [t["result"]["loss"] for t in trials.trials]
    i_best = int(np.argmin(losses))
    rmse_best   = losses[i_best]
    params_best = space_eval(space, best)

    print(f"▶ {model_name} best RMSE: {rmse_best:.4f} with params {params_best}")

    if rmse_best < best_overall["rmse"]:
        best_overall.update({
            "model": model_name,
            "rmse": rmse_best,
            "params": params_best
        })

# 4) Overall winner
print("\n=== Overall Best Model ===")
print(f"{best_overall['model']} → RMSE={best_overall['rmse']:.4f}")
print("Params:", best_overall["params"])
