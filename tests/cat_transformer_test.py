import pandas as pd
import numpy as np
from ml_project.features.transformers import CatTransformer


def test_simple_pandas():
    train = pd.DataFrame({"A": ["a", "b", "c", "d"]})
    ans = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ctran = CatTransformer(10)
    trans_train = ctran.fit_transform(train)

    assert ctran.cat_feats == ["A"]
    assert np.array_equal(trans_train.to_numpy(), ans)


def test_pandas_to_numpy():
    train = pd.DataFrame({"A": [1, 2, 3, 4]})
    test = np.array([[1], [2], [3], [4]])
    ans = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ctran = CatTransformer(10).fit(train)
    trans_test = ctran.transform(test)

    assert ctran.cat_feats == ["A"]
    assert np.array_equal(trans_test.to_numpy(), ans)


def test_numpy_to_pandas():
    test = pd.DataFrame({"A": [1, 2, 3, 4]})
    train = np.array([[1], [2], [3], [4]])
    ans = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ctran = CatTransformer(10).fit(train)
    trans_test = ctran.transform(test)

    assert np.array_equal(trans_test, ans)


def test_two_cols_pandas():
    train = pd.DataFrame(
        {
            "A": ["a", "b", "c", "d"],
            "B": ["a", "b", "c", "c"],
        },
    )
    ans = np.array([["a", 0, 0], ["b", 1, 0], ["c", 0, 1], ["d", 0, 1]])
    ctran = CatTransformer(3)
    trans_train = ctran.fit_transform(train).to_numpy()
    assert ctran.cat_feats == ["B"]
    assert np.array_equal(trans_train.astype("str"), ans)


def test_less_uniques_pandas():
    train = pd.DataFrame({"A": ["a", "b", "c", "d"]})
    test = pd.DataFrame({"A": ["a", "b", "c", "a"]})
    ans = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]
    ctran = CatTransformer(10)
    ctran.fit(train)
    trans_test = ctran.transform(test)

    assert ctran.cat_feats == ["A"]
    assert np.array_equal(trans_test.to_numpy(), ans)


def test_simple_numpy():
    train = np.array([[0, 0], [1, 1], [2, 2], [2, 3]])
    ctran = CatTransformer(3)
    ans = [[0, 0, 0], [1, 0, 1], [0, 1, 2], [0, 1, 3]]

    trans_train = ctran.fit_transform(train)

    assert ctran.cat_feats == [0]
    assert np.array_equal(ans, trans_train)
