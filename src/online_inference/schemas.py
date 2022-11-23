import numpy as np
from pydantic import BaseModel, validator, Field
from pydantic import ValidationError
from pydantic.error_wrappers import ErrorWrapper
from typing import Union, List


numeric = Union[int, float]
ARGS_COUNT = 13


class Patient(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError(
                "You should specify all arguments as non-keyword or keyword, not both."
            )

        if args:
            kwargs = {}
            for i, field in enumerate(self._fields_names()):
                kwargs[field] = args[i]

        return super().__init__(**kwargs)

    def form_sample(self) -> np.ndarray:
        sample = []

        for field in self._fields_names():
            sample.append(getattr(self, field))

        return np.array(sample)

    @validator("age")
    def normal_age(cls, fval: int):
        if fval <= 0 or fval > 130 or not (isinstance(fval, int)):
            raise ValueError("Age is an integer in (0, 130]")
        return fval

    @validator("trestbps")
    def normal_pressure(cls, fval: int):
        if fval < 50 or fval > 300 or not (isinstance(fval, int)):
            raise ValueError("Blood pressure is an integer in [50, 300]")
        return fval

    @validator("chol")
    def normal_chol(cls, fval: int):
        if fval < 100 or fval > 600 or not (isinstance(fval, int)):
            raise ValueError("Serum estoral is an integer in [100, 600] mg/dl")
        return fval

    @validator("thalach")
    def normal_thalach(cls, fval: int):
        if fval < 50 or fval > 250 or not (isinstance(fval, int)):
            raise ValueError("Maximum achieved heart rate is an integer in [50, 250]")
        return fval

    @validator("oldpeak")
    def normal_oldpeak(cls, fval: int):
        if fval < 0 or fval > 10 or not (isinstance(fval, float)):
            raise ValueError("ST is a float in [0, 10]")
        return fval

    @validator("sex")
    def check_sex(cls, fval: int):
        if fval not in [0, 1]:
            raise ValueError("Sex should be 0 (female), 1 (male)")
        return fval

    @validator("cp")
    def chest_pain_type(cls, fval: int):
        if fval not in [0, 1, 2, 3]:
            raise ValueError("Chest pain type should be one of 0-3 type.")
        return fval

    @validator("fbs")
    def fast_blood_sugar_ind(cls, fval: int):
        if fval not in [0, 1]:
            raise ValueError(
                "Fasting blood sugar should be 0 or 1 (<=120 mg/dl or not)"
            )
        return fval

    @validator("restecg")
    def eeg_results(cls, fval: int):
        if fval not in [0, 1, 2]:
            raise ValueError("Resting electrocardiographic result should be 0, 1 or 2")
        return fval

    @validator("exang")
    def angina_results(cls, fval: int):
        if fval not in [0, 1]:
            raise ValueError("induced angina indicator (1 = yes; 0 = no)")
        return fval

    @validator("slope")
    def slope_type(cls, fval: int):
        if fval not in [0, 1, 2]:
            raise ValueError(
                "The slope of the peak exercise ST segment should be 0, 1, or 2"
            )
        return fval

    @validator("ca")
    def number_of_vessels(cls, fval: int):
        if fval not in [0, 1, 2, 3]:
            raise ValueError("Number of major vessels (0-3)")
        return fval

    @validator("thal")
    def defect_type(cls, fval: int):
        if fval not in [0, 1, 2]:
            raise ValueError(
                "0 = normal; 1 = fixed defect; 2 = reversable defect and the label"
            )
        return fval

    def _fields_names(self):
        return Patient.__fields__.keys()


class PatientFromList(BaseModel):
    data_mat: List[List[numeric]]

    def create_patients(self) -> List[Patient]:
        errors = []
        patients = []
        for i, line in enumerate(self.data_mat):
            try:
                cpat = Patient(*line)
                patients.append(cpat)
            except ValidationError as ve:
                errors.append(ErrorWrapper(ve, loc=str(i)))

        if errors:
            raise ValidationError(errors, model=Patient)
        return patients

    @validator("data_mat")
    def len_check(cls, data_mat: List[List[numeric]]):
        errors = []
        for i, line in enumerate(data_mat):
            if len(line) != ARGS_COUNT:
                errors.append(
                    ErrorWrapper(
                        ValueError(
                            f"line has wrong number of arguments: {len(line)} != {ARGS_COUNT}",
                        ),
                        loc=str(i),
                    )
                )
        if errors:
            raise ValidationError(errors, model=PatientFromList)
        return data_mat


class ResponseTarget(BaseModel):
    pos: int
    target: int
    proba: float = Field(ge=0, le=1)
