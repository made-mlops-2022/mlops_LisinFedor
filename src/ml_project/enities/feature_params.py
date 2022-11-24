from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureParams(object):
    """Params for feature pipline."""

    target_col: str
    features_to_drop: Optional[List[str]]
    categorical_features_max_uniqs: Optional[int] = field(default=5)
    numeric_features_min_uniqs: Optional[int] = field(default=5)
    create_dummy: bool = field(default=True)
    use_scaler: bool = field(default=True)
    scaler: str = field(default="MinMaxScaler")
