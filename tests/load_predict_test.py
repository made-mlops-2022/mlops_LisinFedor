from math import isclose
from pathlib import Path
from ml_project.models.model_fit_predict import load_local_model
from online_inference.schemas import Patient
from online_inference.model_load import make_prediction


def test_consistency(assets: Path):
    patients = [
        Patient(*[66, 0, 3, 178, 228, 1, 0, 165, 1, 1.0, 1, 2, 2]),
        Patient(*[59, 1, 3, 140, 177, 0, 0, 162, 1, 0.0, 0, 1, 2]),
        Patient(*[41, 1, 1, 135, 203, 0, 1, 132, 0, 0.0, 1, 0, 1]),
    ]

    model = load_local_model(assets / "LogisticRegression_experiment")
    p1 = make_prediction(model, patients[:1])
    p2 = make_prediction(model, patients[:2])
    p3 = make_prediction(model, patients[:3])
    assert p1[0].target == p2[0].target == p3[0].target
    assert p2[1].target == p3[1].target

    assert isclose(p1[0].proba, p2[0].proba)
    assert isclose(p2[0].proba, p3[0].proba)
