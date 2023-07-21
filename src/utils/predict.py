from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly

# Some regression-alike functions
# * I did not test them

np.random.seed(123)


def exponential_regression(data: pd.DataFrame, X_column: str, y_column: str, weights: np.array = None, return_model: bool = False) -> tuple[np.array, float]:
    X = data[X_column].values.reshape(-1, 1)
    y = data[y_column].values.reshape(-1, 1)
    transformer = FunctionTransformer(np.log, validate=True)
    y_transformed = transformer.fit_transform(y)

    lr = LinearRegression()
    lr.fit(X, y_transformed, sample_weight=weights)
    y_pred = lr.predict(X)
    coef = lr.coef_
    r2 = lr.score(X, y_transformed, sample_weight=weights)
    if return_model:
        return lr
    else:
        return y_pred, coef, r2


def polynomial_regression(data: pd.DataFrame,
                          X_column: str,
                          y_column: str,
                          power: int,
                          weights: np.array = None,
                          X_new: np.array = None,
                          X_start: int = 0,
                          X_end: int = 40,
                          X_num: int = 100):
    # !: Weights are not used in this function
    X = data[X_column].squeeze().T
    y = data[y_column].squeeze().T
    coef = poly.polyfit(X, y, power)

    if X_new is None:
        X_new = np.linspace(X_start, X_end, num=X_num)

    f = poly.polyval(X_new, coef)

    return X_new, f


def linear_regression(data: pd.DataFrame, X_column: str, y_column: str, weights: np.array = None, return_model: bool = False) -> tuple[np.array, float, float]:
    '''Do a linear regression on the data and return the predicted values, the coefficient and the r2 score.'''
    X = data[X_column].values.reshape(-1, 1)
    y = data[y_column].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y, sample_weight=weights)
    y_pred = lr.predict(X)
    coef = lr.coef_
    r2 = lr.score(X, y, sample_weight=weights)
    if return_model:
        return lr
    else:
        return y_pred, coef, r2
