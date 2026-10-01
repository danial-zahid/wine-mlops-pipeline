import time
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from src.data import load_and_validate_data


def test_model_quality_gate():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    _, X_test, _, y_test = load_and_validate_data()
    champion_uri = "models:/WineClassifier@champion"
    model = mlflow.sklearn.load_model(champion_uri)
    start_time = time.perf_counter()
    preds = model.predict(X_test)
    batch_latency_ms = (time.perf_counter() - start_time) * 1000
    assert batch_latency_ms <= 30.0
    assert f1_score(y_test, preds, average="macro") >= 0.88
    assert set(preds).issubset({0, 1, 2})
