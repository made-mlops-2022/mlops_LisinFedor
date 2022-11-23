import pytest
from online_inference.schemas import Patient
from pydantic.error_wrappers import ValidationError


def test_args_init():
    args = [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0, 0]
    p = Patient(*args)
    ans_p = Patient(
        age=69,
        sex=1,
        cp=0,
        trestbps=160,
        chol=234,
        fbs=1,
        restecg=2,
        thalach=131,
        exang=0,
        oldpeak=0.1,
        slope=1,
        ca=1,
        thal=0,
    )

    assert p == ans_p


def test_validators():
    with pytest.raises(ValidationError, match=r".* Sex should be .*"):
        args = [69, 3, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0, 0]
        Patient(*args)
