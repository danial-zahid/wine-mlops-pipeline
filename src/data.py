# Integrated Model Tuning Configuration (Resolved)
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_and_validate_data():
    data = load_wine(as_frame=True)
    X = data.data
    y = data.target
    if X.isnull().sum().sum() != 0:
        raise ValueError("Missing values detected.")
    if X.shape[1] != 13:
        raise ValueError("Expected 13 features.")
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
