from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator,
                   X: np.ndarray,
                   y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    ids = np.arange(X.shape[0])

    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)

    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv
    #
    # # split to folds
    # X_split = np.array_split(X, cv)
    # y_split = np.array_split(y, cv)
    #
    # validation_score = 0
    # train_score = 0
    #
    # for i in range(cv):
    #     # Get the i'th fold
    #     train_X, train_y = np.concatenate(X_split[:i]+ X_split[i+1:], axis=0),  np.concatenate(y_split[:i] + y_split[i+1:], axis=0)
    #     test_X, test_y = X_split[i], y_split[i]
    #
    #     estimator.fit(train_X, train_y)
    #
    #     train_score += scoring(train_y, estimator.predict(train_X))
    #     validation_score += scoring(test_y, estimator.predict(test_X))
    #
    # return train_score/(cv), validation_score/(cv)
