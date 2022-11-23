import os

import mlflow
import numpy as np
import pandas as pd
from typing import List
from dotenv import load_dotenv

from online_inference.schemas import Patient, ResponseTarget

load_dotenv()


MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_STAGE = os.environ.get("MODEL_STAGE")
MODEL_VERSION = os.environ.get("MODEL_VERSION")
MLFLOW_URI = os.environ.get("MLFLOW_URI")


def get_model():
    if MODEL_VERSION:
        url = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    else:
        url = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.sklearn.load_model(url)


def make_prediction(model, patients: List[Patient]) -> List[ResponseTarget]:
    df = _request_to_pd(patients)
    return _form_prediction(model, p_data=df)


def _request_to_pd(patients: List[Patient]) -> pd.DataFrame:
    col_names = patients[0]._fields_names()
    p_data = []
    for patient in patients:
        p_data.append(patient.form_sample())

    return pd.DataFrame(p_data, columns=col_names).convert_dtypes()


def _form_prediction(model, p_data: pd.DataFrame) -> List[ResponseTarget]:
    predictions = []
    preds = model.predict(p_data)
    probs = model.predict_proba(p_data)
    probs = probs[np.arange(len(preds)), preds]
    for i, tar_prob in enumerate(zip(preds, probs)):
        response = ResponseTarget(pos=i, target=tar_prob[0], proba=tar_prob[1])
        predictions.append(response)

    return predictions
