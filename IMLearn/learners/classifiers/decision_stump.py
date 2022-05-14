from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np


from IMLearn.metrics import loss_functions


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # init
        loss_rate = np.inf
        num_features = len(X[0])

        # find the feature with the optimal loss
        for j in range(num_features):
            feature = X[:, j]
            for sign in (-1, 1):
                cur_threshold, cur_loss = self._find_threshold(feature, y, sign)
                if cur_loss < loss_rate:
                    loss_rate, self.j_, self.threshold_ = cur_loss, j, cur_threshold
                    self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.empty(X.shape[0])
        y_pred[X[:, self.j_] < self.threshold_] = -self.sign_
        y_pred[X[:, self.j_] >= self.threshold_] = self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]
        values = values.reshape(-1, 1)

        all_thresholds = np.concatenate([np.array([-np.inf]).reshape(1, 1), values, np.array([np.inf]).reshape(1, 1)])

        # loss when theta is initiated in -inf
        initial_threshold_loss = np.sum(np.abs(labels[np.sign(labels) == sign]))
        losses_by_threshold = np.append(initial_threshold_loss, initial_threshold_loss - np.cumsum(labels * sign))
        min_loss_idx = np.argmin(losses_by_threshold)

        return all_thresholds[min_loss_idx], losses_by_threshold[min_loss_idx]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return loss_functions.misclassification_error(y, y_pred)


if __name__ == '__main__':
    np.random.seed(0)
    # X = np.random.uniform(-20, 20, 10)
    # Y = np.random.uniform(-20,20,10)
    # D = np.ones(10)
    # sign = 1
    # Y[Y>0] = 1
    # Y[Y < 0] = -1
    # Des = DecisionStump()
    # print(Des._find_threshold(X,Y, sign))
