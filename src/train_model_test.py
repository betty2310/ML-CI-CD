import os
import pickle
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train_model import train_and_save_model


class TestModelTraining:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        os.makedirs("models", exist_ok=True)
        yield

    def test_train_model_success(self):
        accuracy = train_and_save_model()
        assert accuracy is not None
        assert isinstance(accuracy, (float, np.floating))
        assert 0 <= accuracy <= 1

    def test_model_accuracy_threshold(self):
        accuracy = train_and_save_model()
        assert accuracy >= 0.90, (
            f"Model accuracy {accuracy:.2%} is below 90% threshold "
        )

    def test_model_file_created(self):
        train_and_save_model()
        assert os.path.exists("models/model.pkl")
        assert os.path.getsize("models/model.pkl") > 0

    def test_model_can_be_loaded(self):
        train_and_save_model()
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
            assert isinstance(model, RandomForestClassifier)
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_proba")
