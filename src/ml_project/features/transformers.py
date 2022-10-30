"""Transfrmers for numeric and categotical features."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


log = logging.getLogger(__name__)

DataT = Union[pd.DataFrame, np.ndarray]


class CatTransformer(BaseEstimator, TransformerMixin):
    """Transform categorical features to dummy vectors."""

    def __init__(self, max_cat_num: int) -> None:
        """
        Transform categorical features to dummy vectors.

        If the number of unique values of some feature is less or equal
        to `max_cat_num` this feature will be used to create a dummy vector.

        Args:
            max_cat_num (int): max number of unique elements.
        """
        self.max_cat_num = max_cat_num
        self._cat_feat_names: List[str] = []
        self._enc_feat_names: List[str] = []
        self._cat_feat_id: List[int] = []
        self._enc_num_id: Optional[int] = None
        self._ohe: Optional[ColumnTransformer] = None
        super().__init__()

    def fit(self, x_data: DataT, *args) -> CatTransformer:
        """Found and save position of categorical features.

        Args:
            x_data (DataT): numpy or pandas data. Must not contain a target.

        Returns:
            CatTransformer: fitted transformer.
        """
        if isinstance(x_data, pd.DataFrame):
            self._find_in_pd(x_data)
            transformed_x = self._transform_pd(x_data)
            self._enc_feat_names = transformed_x.columns.to_list()
        elif isinstance(x_data, np.ndarray):
            self._find_in_np(x_data)
            self._ohe = self._create_np_trans()
            self._enc_num_id = self._ohe.fit_transform(x_data).shape[1]

        self._check_cats()
        return self

    def transform(self, x_data: DataT, *args):
        if isinstance(x_data, pd.DataFrame):
            trans_data = self._transform_pd(x_data)
            return self._check_transformation(trans_data)
        elif isinstance(x_data, np.ndarray):
            trans_data = self._transform_np(x_data)
            return self._check_transformation(trans_data)

        raise TypeError(
            "`x_data` can be pandas DataFrame or Numpy, not {tp}".format(
                tp=type(x_data),
            ),
        )

    @property
    def cat_feats(self) -> Union[List[str], List[int]]:
        """Categorical features founded in fit."""
        if not self._cat_feat_names:
            if not self._cat_feat_id:
                raise AttributeError("You need to fit transformer first.")
            return self._cat_feat_id
        return self._cat_feat_names

    def _find_in_pd(self, x_data: pd.DataFrame):
        for col_name in x_data.columns:
            if len(x_data[col_name].unique()) <= self.max_cat_num:
                self._cat_feat_names.append(col_name)

    def _find_in_np(self, x_data: np.ndarray):
        for col_n in range(x_data.shape[1]):
            col = x_data[:, col_n]
            if len(np.unique(col)) <= self.max_cat_num:
                self._cat_feat_id.append(col_n)

    def _check_cats(self):
        if (not self._cat_feat_id) and (not self._cat_feat_names):
            err_txt = (
                "Can't fit CatTransformer. No cat features in data.\n"
                "Try to change `max_cat_num` parameter."
            )
            raise ValueError(err_txt)

    def _create_np_trans(self) -> ColumnTransformer:
        ohe = OneHotEncoder(
            drop="first",
            sparse=False,
        )

        return ColumnTransformer(
            transformers=[
                ("Cattrans", ohe, self._cat_feat_id),
            ],
            remainder="passthrough",
        )

    def _transform_pd(self, x_data: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(
            x_data,
            columns=self._cat_feat_names,
            drop_first=True,
        )

    def _transform_np(self, x_data: np.ndarray) -> np.ndarray:
        if self._ohe is None:
            raise AttributeError("You need to fit transformer first.")
        return self._ohe.transform(x_data)

    def _check_transformation(self, trans_data: DataT) -> DataT:
        if self._ohe is not None:
            if trans_data.shape[0] != self._enc_num_id:
                log.warning(
                    "Number of columns in your input after transformation less,"
                    "then number of columns in fitted data after transformation",
                )
        elif isinstance(trans_data, pd.DataFrame):

            if self._enc_feat_names != trans_data.columns.to_list():
                trans_data = trans_data.reindex(columns=self._enc_feat_names).fillna(0)

        return trans_data
