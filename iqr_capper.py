
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    # Winsorise chosen columns at Q1 - k*IQR / Q3 + k*IQR. Bounds are learned in fit() (training data only).
    def __init__(self, cols=None, k=1.5):
        self.cols = cols
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.bounds_ = {}
        for c in (self.cols if self.cols is not None else self.columns_):
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[c] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_ if input_features is None else input_features, dtype=object)
