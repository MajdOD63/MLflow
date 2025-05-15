# Student Performance MLflow Pipeline

An end-to-end machine-learning system using MLflow to track experiments, train & tune models, deploy the best model to production, and monitor its performance over time.

## Project Overview

This project predicts student exam performance based on various factors including study habits, lifestyle choices, and demographic information. It uses:

- **MLflow** for experiment tracking, model registry, and serving
- **Scikit-learn** for model development
- **Pandas/NumPy** for data processing

## Project Structure

```
project/
├─ data/
│  ├─ student_habits_performance.csv
│  ├─ input_features.csv
   └─ new_student_habits.csv 
├─ src/
│  ├─ preprocess.py  # Data preprocessing pipeline
│  ├─ train.py       # Train baseline models
│  ├─ tune.py        # Hyperparameter tuning
│  ├─ promote_best.py # Promote best model to production
│  ├─ predict.py     # Make predictions using the model
│  └─ monitor.py     # Model performance monitoring
├─ models/
├─ mlruns/
├─ README.md
└─ requirements.txt
```

## Setup

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt 
   ```

2. **Run MLflow server**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

3. **Train Baseline Models**
   ```bash
   python src/train.py
   ```

4. **Hyperparameter Tuning** 
   ```bash
   python src/tune.py
   ```

5. **Promote Best Model to Production**
   ```bash
   python src/promote_best.py
   ```

## Model Serving and Predictions

### Using MLflow CLI for Predictions

You can use the MLflow CLI to make predictions using the production model:

```bash
    mlflow models serve -m "models:/StudentPerformanceModel/Production" -p 1234 --no-conda 
```

### Input Data Format

The input CSV file should contain the following columns in this exact order:

```
student_id,age,gender,study_hours_per_day,social_media_hours,netflix_hours,part_time_job,attendance_percentage,sleep_hours,diet_quality,exercise_frequency,parental_education_level,internet_quality,mental_health_rating,extracurricular_participation,exam_score
```

Example row:
```
S1000,23,Female,0.0,1.2,1.1,No,85.0,8.0,Fair,6,Master,Average,8,Yes,56.2
```

Note: For prediction input, the `exam_score` column will be ignored but should be included for format compatibility.

### Using Python Code for Predictions

```python
    python src/predict.py
```

This returns prediction in a csv file

## Monitoring

To monitor the model's performance over time:

```bash
python src/monitor.py
```

This tracks model drift and performance metrics on new data.

