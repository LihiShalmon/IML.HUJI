from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ , n_k = np.unique(y,return_counts=True)
        n_samples, n_features = X.shape[0],X.shape[1]

        self.mu_ = np.zeros((len(self.classes_),n_features))
        self.vars_ = np.zeros((len(self.classes_),n_features))
        cov = []

        for idx in range(self.classes_.size):
            idx_in_class = (y == self.classes_[idx])
            self.mu_[idx] = np.mean(X[idx_in_class], axis=0)

            vect = X[idx_in_class] - self.mu_[idx]
            cov.append(np.diag(np.power(vect, 2).mean(axis = 0)))
            for feature in range(n_features):
                self.vars_[idx][feature] = cov[idx][feature][feature]
        self.vars_ = np.array(cov)
        self.cov_ = self.vars_
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
        likelihood = np.ndarray((len(X), len(self.classes_)))

        for k in range(len(self.classes_)):
            cov = self.vars_[k]

            const_normal_distribution = 1 / np.sqrt(
                np.power(2 * np.pi, len(X)) * np.linalg.det(cov))

            vec_for_exp = np.diag((X - self.mu_[k]) @ np.linalg.inv(cov) @
                          np.transpose((X - self.mu_[k])))

            likelihood[:, k] = self.pi_[k] * const_normal_distribution * np.exp((-1 / 2) * vec_for_exp)

        return likelihood

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
