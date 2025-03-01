from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ , n_k = np.unique(y,return_counts=True)
        n_samples, n_features = X.shape[0],X.shape[1]

        self.mu_ = []

        for idx in range(self.classes_.size):
            self.mu_.append(np.mean(X[y==idx], axis=0))

        self.mu_ = np.array(self.mu_)
        y_class_ind = np.searchsorted(self.classes_ , y)
        mu_vector = self.mu_[y_class_ind]
        self.cov_ = np.matmul((X - mu_vector).T, (X - mu_vector))/(n_samples - self.classes_.size)
        self._cov_inv = inv(self.cov_)
        self.pi_ = n_k / n_samples
        self.fitted_ = True

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
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        y_hat = np.zeros(n_samples)
        likelihoods = self.likelihood(X)

        for i in range(n_samples):
            y_hat[i] = self.classes_[np.argmax(likelihoods[i,:])]

        return y_hat

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape[0], X.shape[1]
        likelihoods =[]

        for i in range(n_samples):
            likelihood_over_classes = []

            # look for the label which maximizes probability for this sample
            for idx in range(self.classes_.size):
                a_k = self._cov_inv @ self.mu_[idx]
                b_k = np.log(self.pi_[idx]) - 0.5 * (self.mu_[idx] @ self._cov_inv @ self.mu_[idx])
                likelihood_over_classes.append((a_k.T @ X[i,:] + b_k))

            likelihoods.append(likelihood_over_classes)

        return  np.array(likelihoods)


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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
