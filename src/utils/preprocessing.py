import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, indices_1, indices_2, feature_range_1=(-1, 1), feature_range_2=(0, 1)):
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.feature_range_1 = feature_range_1
        self.feature_range_2 = feature_range_2
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.min_ = None
    
    def fit(self, X, y=None):
        # Compute the minimum and maximum to be used for later scaling
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        # Initialize scale and min arrays
        self.data_range_ = np.ones(X.shape[1])
        self.min_ = np.zeros(X.shape[1])

        # Set scale and min for indices_1 based on maximum magnitude
        max_abs_1 = np.maximum(np.abs(self.data_min_[self.indices_1]), np.abs(self.data_max_[self.indices_1]))
        self.data_range_[self.indices_1] = 1 / max_abs_1
        self.min_[self.indices_1] = 0  # Centering at zero
        
        # Set scale and min for indices_2
        range_2 = self.feature_range_2[1] - self.feature_range_2[0]
        self.data_range_[self.indices_2] = range_2 / (self.data_max_[self.indices_2] - self.data_min_[self.indices_2])
        self.min_[self.indices_2] = self.feature_range_2[0] - self.data_min_[self.indices_2] * self.data_range_[self.indices_2]
        
        return self

    def transform(self, X):
        return X * self.data_range_ + self.min_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    

