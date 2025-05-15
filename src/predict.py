# predict.py
import pandas as pd
import mlflow.pyfunc
from preprocess import DataPreprocessor  # your preprocess.py

def main():
    # 1) Load the Production model
    model = mlflow.pyfunc.load_model("models:/StudentPerformanceModel/Production")

    # 2) Load the preprocessor (you should have one saved for the best model)
    #    Adjust the path if you named it differently.
    preproc = DataPreprocessor.load_preprocessor("models/preprocessor_RandomForest.joblib")

    # 3) Read raw CSV
    df_raw = pd.read_csv("data/new_student_habits.csv")

    # 4) Preprocess (drops/casts/encodes/etc.)
    #    preprocess_data returns (X, y) but y is ignored at inference
    X, _ = preproc.preprocess_data(df_raw, is_training=False)

    # 5) Predict
    preds = model.predict(X)

    # 6) Save predictions
    out = pd.DataFrame({"prediction": preds})
    out.to_csv("output_predictions.csv", index=False)
    print("👉 Wrote output_predictions.csv")

if __name__ == "__main__":
    main()
