import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ml_project.features.transformers import ScalerTransformer


def test_simple_pd():
    train = pd.DataFrame({"A": [0, 2, 4, 8]})
    test = pd.DataFrame({"A": [0., 0.25, 0.5, 1.]})
    test_scaler = MinMaxScaler()
    scale_trans = ScalerTransformer(test_scaler, 1)
    trans_train = scale_trans.fit_transform(train)
    assert np.isclose(trans_train["A"].to_numpy(), test["A"].to_numpy()).all()


def test_np_skip_str():
    train = np.array([[0, 0], [2, 1], [4, 0], [8, 1]])
    test = np.array([[0, 0], [0.25, 1], [0.5, 0], [1., 1]])
    test_scaler = MinMaxScaler()
    scale_trans = ScalerTransformer(test_scaler, 3)
    trans_train = scale_trans.fit_transform(train)
    assert np.isclose(trans_train, test).all()
