import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

import pandas as pd
import numpy as np
import requests
import json
from sklearn.metrics import mean_squared_error
from datetime import datetime
from preprocess import DataPreprocessor

mlflow.set_experiment("model_monitoring")

# Connect to the model being served, not production
SERVED_MODEL_ENDPOINT = "http://localhost:1234/invocations"

def check_performance(new_data_csv):
    # Load and preprocess data
    df = pd.read_csv(new_data_csv)
    preproc = DataPreprocessor()
    X, y = preproc.preprocess_data(df, is_training=False)
    
    # Convert to format expected by MLflow serving
    input_data = X.to_dict(orient='split')
    
    try:
        # Make predictions using the served model
        response = requests.post(
            SERVED_MODEL_ENDPOINT,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"dataframe_split": input_data})
        )
        
        # Debug the response
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")  # Print the first 200 chars
        
        if response.status_code != 200:
            print(f"Error from model server: {response.text}")
            return
        
        # Parse predictions properly
        response_data = response.json()
        
        # Check if the response is a dictionary with a 'predictions' key
        if isinstance(response_data, dict) and 'predictions' in response_data:
            predictions = response_data['predictions']
        else:
            # Assume direct array of predictions
            predictions = response_data
        
        # Convert to floats
        predictions = [float(pred) for pred in predictions]
        
        # Display predictions vs actuals for each row
        for i, (pred, actual) in enumerate(zip(predictions, y)):
            print(f"Row {i+1}: pred={pred:.2f}, actual={actual:.2f}")
        
        # Calculate and log RMSE
        rmse = np.sqrt(mean_squared_error(y, predictions))
        print(f"\nLogged live RMSE: {rmse:.2f}")
        
        # Log to MLflow for tracking
        with mlflow.start_run(run_name=f"live_monitor_{datetime.utcnow().isoformat()}"):
            mlflow.log_metric("rmse", rmse)
            # Log individual predictions for later analysis
            for i, (pred, actual) in enumerate(zip(predictions, y)):
                mlflow.log_metric(f"pred_{i}", pred)
                mlflow.log_metric(f"actual_{i}", actual)
                mlflow.log_metric(f"error_{i}", abs(pred - actual))
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    check_performance("data/new_student_habits.csv")
