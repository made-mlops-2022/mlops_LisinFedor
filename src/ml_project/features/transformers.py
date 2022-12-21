"""Transfrmers for numeric and categotical features."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


logger = logging.getLogger(__name__)

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
        logger.info(
            "Feature is categorical if number of unique vals less then %d.",  # noqa: WPS323
            max_cat_num,
        )
        self.max_cat_num = max_cat_num
        self._cat_feat_names: List[str] = []
        self._enc_feat_names: List[str] = []
        self._cat_feat_id: List[int] = []
        self._enc_num_id: Optional[int] = None
        self._ohe: Optional[ColumnTransformer] = None
        self._pd_feats: Optional[List[str]] = None
        self._np_feats_count: Optional[int] = None
        super().__init__()

    def fit(self, x_data: DataT, *args) -> CatTransformer:
        """Found and save position of categorical features.

        Args:
            x_data (DataT): numpy or pandas data. Must not contain a target.
            *args: args.

        Returns:
            CatTransformer: fitted transformer.
        """
        if isinstance(x_data, pd.DataFrame):
            self._pd_feats = x_data.columns.to_list()
            self._find_in_pd(x_data)
            transformed_x = self._transform_pd(x_data)
            self._enc_feat_names = transformed_x.columns.to_list()
        elif isinstance(x_data, np.ndarray):
            self._np_feats_count = x_data.shape[1]
            self._find_in_np(x_data)
            self._ohe = self._create_np_trans()
            self._enc_num_id = self._ohe.fit_transform(x_data).shape[1]

        self._check_cats()
        return self

    def transform(self, x_data: DataT, *args):
        formated_data = self._format_data(x_data)
        if isinstance(formated_data, pd.DataFrame):
            trans_data = self._transform_pd(formated_data)
            return self._check_transformation(trans_data)
        elif isinstance(formated_data, np.ndarray):
            trans_data = self._transform_np(formated_data)
            return self._check_transformation(trans_data)

        raise TypeError(
            "`x_data` can be pandas DataFrame or Numpy, not {tp}".format(
                tp=type(formated_data),
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

    def _format_data(self, x_data: DataT):
        """Change data type pandas <-> numpy"""
        if isinstance(x_data, pd.DataFrame):
            if self._pd_feats is not None:
                return x_data
            return x_data.to_numpy()
        if isinstance(x_data, np.ndarray):
            if self._np_feats_count is not None:
                return x_data
            return pd.DataFrame(x_data, columns=self._pd_feats)

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
        )

    def _transform_np(self, x_data: np.ndarray) -> np.ndarray:
        if self._ohe is None:
            raise AttributeError("You need to fit transformer first.")
        return self._ohe.transform(x_data)

    def _check_transformation(self, trans_data: DataT) -> DataT:
        if self._ohe is not None:
            if trans_data.shape[0] != self._enc_num_id:
                logger.warning(
                    "Number of columns in your input after transformation less,"
                    "then number of columns in fitted data after transformation",
                )
        elif isinstance(trans_data, pd.DataFrame):

            if self._enc_feat_names != trans_data.columns.to_list():
                old_cols = str(self._enc_feat_names)
                cols = str(trans_data.columns.to_list())
                logger.warning("Number of columns after conversion does not match.")
                logger.warning("Expected columns: %s", old_cols)  # noqa: WPS323
                logger.warning("Founded: %s", cols)  # noqa: WPS323
                logger.warning("Missing columns added and filled with 0.")
                trans_data = trans_data.reindex(columns=self._enc_feat_names).fillna(0)

        return trans_data


class ScalerTransformer(BaseEstimator, TransformerMixin):
    """Transform numeric features unsing scaler."""

    def __init__(self, scaler: TransformerMixin, min_numeric: int) -> None:
        """Init transformer.

        Args:
            scaler (TransformerMixin): scaler for scaling features.
            min_numeric (int): min number if unique values.
                If feature has more uniq values than min_numeric,
                this feature will be rescaled.
        """
        self.scaler = scaler
        self.min_numeric = min_numeric
        self._cols: List[Union[str, int]] = []
        self._transformer: Optional[ColumnTransformer] = None
        super().__init__()

    def fit(self, x_data: DataT, *args) -> ScalerTransformer:
        fit_data = x_data.copy()
        if isinstance(x_data, pd.DataFrame):
            fit_data = x_data.to_numpy()
        self._find_numeric(fit_data)
        self._transformer = ColumnTransformer(
            transformers=[
                ("scaler", self.scaler, self._cols),
            ],
        )
        self._transformer.fit(fit_data)

        return self

    def transform(self, x_data: DataT, *args) -> DataT:
        cols_names = None
        if isinstance(x_data, np.ndarray):
            tr_data = x_data.copy()
        elif isinstance(x_data, pd.DataFrame):
            tr_data = x_data.to_numpy()
            cols_names = x_data.columns
        else:
            raise TypeError(
                "`x_data` can be pandas DataFrame or Numpy, not {tp}".format(
                    tp=type(x_data),
                ),
            )

        if self._transformer is None:
            raise AttributeError("You need to fit transformer first.")
        scaled_cols = self._transformer.transform(tr_data)
        tr_data = tr_data.astype("float")
        tr_data[:, self._cols] = scaled_cols

        if cols_names is not None:
            return pd.DataFrame(tr_data, columns=cols_names)

        return tr_data

    def _find_numeric(self, df: DataT):
        if isinstance(df, pd.DataFrame):
            self._find_numeric_pd(df)
        elif isinstance(df, np.ndarray):
            self._find_numeric_np(df)
        else:
            raise TypeError(
                "`x_data` can be pandas DataFrame or Numpy, not {tp}".format(
                    tp=type(df),
                ),
            )

    def _find_numeric_pd(self, df: pd.DataFrame):
        for col in df.columns:
            uniqs_num = len(df[col].unique())
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                if uniqs_num > self.min_numeric:
                    self._cols.append(col)
            else:
                logger.warning(
                    "Skip column %s. Not a numeric type.",  # noqa: WPS323
                    str(col),
                )

    def _find_numeric_np(self, df: np.ndarray):
        for col_i in range(df.shape[1]):
            uniqs_num = len(np.unique(df[:, col_i]))
            if pd.api.types.is_numeric_dtype(df[:, col_i]):
                if uniqs_num > self.min_numeric:
                    self._cols.append(col_i)
            else:
                raise ValueError("All values in np.array mast be numeric")
