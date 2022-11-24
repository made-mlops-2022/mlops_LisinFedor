from typing import Optional
import pandas as pd
import logging
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from ml_project.enities.feature_params import FeatureParams
from ml_project.features.transformers import CatTransformer, ScalerTransformer


logger = logging.getLogger(__name__)


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]


def build_data_preproc_pipline(params: FeatureParams) -> Optional[Pipeline]:
    """Build pipline for data preprocessing; data should have no target."""

    final_pipe = Pipeline(steps=[])

    if params.create_dummy:
        logger.info("Add CatTransformer to pipeline.")
        final_pipe.steps.append(
            (
                "Create dummy",
                CatTransformer(max_cat_num=params.categorical_features_max_uniqs),
            ),
        )

    if params.use_scaler:
        logger.info("Add %s scaler for numeric features.", params.scaler)
        scaler = getattr(preprocessing, params.scaler)()
        final_pipe.steps.append(
            (
                "Scale numeric",
                ScalerTransformer(
                    scaler, min_numeric=params.numeric_features_min_uniqs
                ),
            ),
        )

    if final_pipe.steps:
        return final_pipe

    return None
