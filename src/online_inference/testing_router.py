from fastapi import APIRouter, Query, Depends, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic.error_wrappers import ValidationError

from online_inference.model_load import get_model, make_prediction
from online_inference import schema_utils
from online_inference.schemas import Patient, ResponseTarget


testing_router = APIRouter()


@testing_router.post("/predict/one", response_model=ResponseTarget)
async def one(
    age: int,
    trestbps: int,
    chol: int,
    thalach: int,
    oldpeak: float,
    sex: str = Query(
        enum=schema_utils.Sex.keys(),
        description="Sex.",
    ),
    cp: str = Query(
        enum=schema_utils.ChestPain.keys(),
        description="Chest pain type.",
    ),
    fbs: str = Query(
        enum=schema_utils.FastingBlood.keys(),
        description="Fasting blood sugar > 120 mg/dl",
    ),
    restecg: str = Query(
        enum=schema_utils.EegResult.keys(),
        description="Resting electrocardiographic results.",
    ),
    exang: str = Query(
        enum=schema_utils.ExInducedAngina.keys(),
        description="Exercise induced angina.",
    ),
    slope: str = Query(
        enum=schema_utils.SlopeSegment.keys(),
        description="The slope of the peak exercise ST segment.",
    ),
    ca: int = Query(
        enum=[0, 1, 2, 3],
        description="Number of major vessels .",
    ),
    thal: str = Query(
        enum=schema_utils.Thal.keys(),
        description="Have no idea what the fuck is this.",
    ),
    model=Depends(get_model),
):
    kwargs = locals().copy()
    kwargs["sex"] = getattr(schema_utils.Sex, sex).value
    kwargs["exang"] = getattr(schema_utils.ExInducedAngina, exang).value
    kwargs["cp"] = getattr(schema_utils.ChestPain, cp).value
    kwargs["fbs"] = getattr(schema_utils.FastingBlood, fbs).value
    kwargs["restecg"] = getattr(schema_utils.EegResult, restecg).value
    kwargs["slope"] = getattr(schema_utils.SlopeSegment, slope).value
    kwargs["thal"] = getattr(schema_utils.Thal, thal).value

    try:
        patient = Patient(**kwargs)
    except ValidationError as ve:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": ve.errors()}),
        )
    return make_prediction(model, [patient])[0]
