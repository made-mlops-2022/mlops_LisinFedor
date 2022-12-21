from enum import IntEnum


class BaseIntEnum(IntEnum):
    @classmethod
    def keys(cls):
        return [el.name for el in cls]


class Sex(BaseIntEnum):
    female: int = 0
    male: int = 1


class ChestPain(BaseIntEnum):
    typical: int = 0
    atypical: int = 1
    non_anginal: int = 2
    asymptomatic: int = 3


class FastingBlood(BaseIntEnum):
    bigger120: int = 1
    smaller120: int = 0


class EegResult(BaseIntEnum):
    normal: int = 0
    abnormal: int = 1
    hypertrophy: int = 2


class ExInducedAngina(BaseIntEnum):
    yes: int = 1
    no: int = 0


class SlopeSegment(BaseIntEnum):
    upsloping: int = 0
    flat: int = 1
    downsloping: int = 2


class Thal(BaseIntEnum):
    normal: int = 0
    fixed: int = 1
    reversable: int = 2