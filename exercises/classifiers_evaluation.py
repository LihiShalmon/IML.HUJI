from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
from IMLearn.utils import  split_train_test
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
            # Load dataset
            filename = r"C:/Users/LIHI/Documents/Uni/yearB/IML/IML.HUJI/datasets/" + f
            X, y = load_dataset(filename)

            # Fit Perceptron and record loss in each fit iteration
            losses = []

            callable_func = lambda p, x1, y1: losses.append(p._loss(X, y))
            perceptron = Perceptron(callback=callable_func, include_intercept=False)

            perceptron._fit(X, y)
            print(losses)
            # Plot figure of loss as function of fitting iteration
            fig3 = go.Figure(go.Scatter(y=losses, mode='markers+lines'),
                             layout=go.Layout(
                                 title='loss as function of fitting iteration'
                                ,xaxis=dict(title=r"iteration number")
                                ,yaxis=dict(title=r"iteration number")))

            fig3.show()




def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')
        #filename = r"C:/Users/Lihi/Documents/Uni/yearB/IML/IML.HUJI/datasets/" + f
        #X, y = load_dataset(filename)

        train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 0.5)

        # Fit models and predict over training set
        gb = GaussianNaiveBayes()
        lda = LDA()

        models = [gb, lda]
        model_names = ["Gaussian Naive Bayes", "LDA"]

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(models):
            # fit
            models[i]._fit(np.array(train_X), np.array(train_y))
            # predict
            pred = models[i]._predict(np.array(test_X))
            test_X = np.array(test_X)
            fig.add_traces(go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                      mode="markers",
                                      showlegend=False,
                                      marker=dict(color= pred, symbol=np.array(test_y), line=dict(color="black", width=1))),
                                      rows=(i // 3) + 1, cols=(i % 3) + 1)

            fig.update_layout(title=rf"$\textbf{{ Decision Boundaries Of Models}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()
        from IMLearn.metrics import accuracy

        # # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()
        #
        # # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()
        #
        # # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    #run_perceptron()
    compare_gaussian_classifiers()
