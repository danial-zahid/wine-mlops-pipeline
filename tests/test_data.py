import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_and_validate_data


def test_data_pipeline():
    X_train, X_test, y_train, y_test = load_and_validate_data()

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 178, f"Expected 178 samples, got {total_samples}"

    assert X_train.shape[1] == 13, f"Expected 13 features, got {X_train.shape[1]}"
    assert X_test.shape[1] == 13, f"Expected 13 features, got {X_test.shape[1]}"

    assert len(set(y_train)) == 3
    assert len(set(y_test)) == 3

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0