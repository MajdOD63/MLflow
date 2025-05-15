import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

from mlflow.tracking import MlflowClient

client = MlflowClient()
exp    = client.get_experiment_by_name("student_performance_tuning")
runs   = client.search_runs(exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best   = runs[0]
run_id = best.info.run_id

# Register & promote
model_uri = f"runs:/{run_id}/model"
mv        = mlflow.register_model(model_uri, "StudentPerformanceModel")
client.transition_model_version_stage(
    name="StudentPerformanceModel",
    version=mv.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Run {run_id} registered as version {mv.version} → PRODUCTION")
