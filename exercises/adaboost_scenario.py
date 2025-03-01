import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):

    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    adabooster = AdaBoost(DecisionStump, n_learners)
    adabooster = adabooster.fit(train_X, train_y)
    loss_test, loss_train = [], []

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    for i in range(1, n_learners):
        loss_test.append(adabooster.partial_loss(test_X, test_y, i))
        loss_train.append(adabooster.partial_loss(train_X, train_y, i))

    fig1 = go.Figure(go.Scatter(x=np.arange(1, n_learners+1, 1), y=loss_test, name="test",
                                mode="lines",
                                showlegend=True,
                                marker=dict(line=dict(color="black", width=1))))
    fig1.add_scatter(x=np.arange(1, n_learners+1, 1), y=loss_train,
                     mode="lines", name="train",
                     showlegend=True,
                     marker=dict(line=dict(color="black", width=1)))
    fig1.update_layout(title="Training and test errors as a function of the number of learners",
                       xaxis_title="Models in use",
                       yaxis_title="Prediction rate")
    fig1.show()

    # Question 2: Plotting decision surfaces

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    decision_surfaces_plot = make_subplots(rows=2, cols=2, subplot_titles=[f"{m} iterations" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, m in enumerate(T):
        decision_surfaces_plot.add_traces([decision_surface(
            lambda df: adabooster.partial_predict(df, m), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=test_y,
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    decision_surfaces_plot.update_layout(title=rf"$\textbf{{ Decision Boundaries Of the model - by comitee size }}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    decision_surfaces_plot.show()

    # Question 3: Decision surface of best performing ensemble
    optimal_ensemble = np.argmin(loss_test)
    if optimal_ensemble.size > 1:
        optimal_ensemble = optimal_ensemble[0]

    fig_opt_ensemble = go.Figure(data=[decision_surface(lambda X:
                                            adabooster.partial_predict(X, n_learners),
                                            lims[0], lims[1],
                                            showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                      mode="markers",
                                      showlegend=False,
                                      marker=dict(color=test_y,
                                                  colorscale=[custom[0],
                                                              custom[-1]],
                                                  line=dict(color="black",
                                                            width=1)))],
                     layout=go.Layout(
                         title=f"Boundary with lowest test error. Ensemble: {optimal_ensemble}. "
                               f" Accuracy:{round(1-loss_test[optimal_ensemble],3)}"))
    fig_opt_ensemble.show()

    # Question 4: Decision surface with weighted samples
    fig_wighted_samples = go.Figure(data=[decision_surface(adabooster.predict,
                                            lims[0], lims[1],
                                            showscale=False),
                           go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                      mode="markers",
                                      showlegend=False,
                                      marker=dict(color=train_y, size= adabooster.D_/np.max(adabooster.D_) * 5,
                                                  colorscale=[custom[0],
                                                              custom[-1]],
                                                  line=dict(color="black",
                                                            width=1)))],
                     layout=go.Layout(title=f"Full ensemble, sample size indicated weight"))
    fig_wighted_samples.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, n_learners=250)
    fit_and_evaluate_adaboost(0.4, n_learners=250)


