import os
import sys
import time
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_and_validate_data


def test_model_quality_gate():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    _, X_test, _, y_test = load_and_validate_data()

    champion_uri = "models:/WineClassifier@champion"
    model = mlflow.sklearn.load_model(champion_uri)

    start_time = time.perf_counter()
    preds = model.predict(X_test)
    end_time = time.perf_counter()

    batch_latency_ms = (end_time - start_time) * 1000
    assert batch_latency_ms <= 30.0, f"Latency gate failed: {batch_latency_ms:.2f}ms > 30ms"

    macro_f1 = f1_score(y_test, preds, average="macro")
    assert macro_f1 >= 0.88, f"F1 metric gate failed: {macro_f1:.4f} < 0.88"

    valid_classes = {0, 1, 2}
    predicted_classes = set(preds)
    assert predicted_classes.issubset(valid_classes), (
        f"Schema integrity failed: Found invalid classes {predicted_classes - valid_classes}"
    )