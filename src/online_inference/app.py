import os
from typing import List
from fastapi import FastAPI, Depends, Body, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic.error_wrappers import ValidationError
from logging import getLogger

from online_inference.model_load import get_model, make_prediction
from online_inference.schemas import Patient, PatientFromList, ResponseTarget


app = FastAPI()
logger = getLogger(__name__)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.post("/predict/json", response_model=ResponseTarget)
async def json(
    patient: Patient = Body(embed=True),
    model=Depends(get_model),
):
    return make_prediction(model, [patient])[0]


@app.post("/predict/many/json", response_model=List[ResponseTarget])
async def many_json(patients: List[Patient], model=Depends(get_model)):
    return make_prediction(model, patients)


@app.post("/predict/many/matrix", response_model=List[ResponseTarget])
async def many_matrix(
    patients: PatientFromList,
    model=Depends(get_model),
):
    try:
        patients_list = patients.create_patients()
    except ValidationError as ve:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": ve.errors()}),
        )

    return make_prediction(model, patients_list)


@app.post("/health")
async def health_check(model=Depends(get_model)):
    try:
        pat = Patient(*[66, 0, 3, 178, 228, 1, 0, 165, 1, 1.0, 1, 2, 2])
        make_prediction(model, [pat])
    except Exception as ex:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=jsonable_encoder({"detail": ex}),
        )
    return {"response": "OK"}
