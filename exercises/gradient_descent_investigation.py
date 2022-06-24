import os

import numpy as np
import pandas as pd
import  plotly.express as px
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    raise NotImplementedError()


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l2_norm = L2(weights=init)
    plot_for_module_step(etas, init, norm = l2_norm, f=L2, name="L2")
    l1_norm =L1(weights=init)
    plot_for_module_step(etas, init, norm = l1_norm, f=L1 , name="L1")



def plot_for_module_step(etas, init, norm, f, name):
    fig_convergence = go.Figure()
    np_min = np.infty

    def callback(solver, w_t, val, grad, t, eta, delta):
        weights_list[t] = w_t
        list_vals[t] = val
        t_fin[-1] = t

    for step in etas:
        if name =="L2":
            norm = L2(weights=init)
        else:
            norm =L1(weights=init)

        t_fin = [0]
        weights_list = np.empty((1001, 2))
        list_vals = np.empty((1001,))
        weights_list[0] = init

        gd = GradientDescent(learning_rate=FixedLR(step), callback=callback)
        gd.fit(f=norm, X=None, y=None)
        fig = plot_descent_path(module=f, descent_path=weights_list[: t_fin[0]+1], title=name)
        fig.show()
        fig_convergence.add_trace(go.Scatter(x=np.arange(t_fin[0]+1), y=list_vals[:t_fin[0]+1], mode="lines",
                                 name="eta = " + str(step)))

    fig_convergence.update_layout(title="{0} norm as function of gradient descent iteration".format(name),
                             legend_title = "Chosen eta")
    fig_convergence.show()
    print("min for "+ name+ " is: "+ str(np.min(list_vals)))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):

    # Plot algorithm's convergence for the different values of gamma
    fig_convergence = go.Figure()
    np_min = np.infty

    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    for gama in gammas:

        def callback(solver, w_t, val, grad, t, eta, delta):
            weights_list[t] = w_t
            list_vals[t] = val
            t_fin[-1] = t

        t_fin = [0]
        weights_list = np.empty((1001, 2))
        list_vals = np.empty((1001,))
        weights_list[0] = init
        norm = L1(weights=init)

        gd = GradientDescent(learning_rate=ExponentialLR(eta, gama), callback=callback)
        gd.fit(f=norm, X=None, y=None)
        # Plot descent path for gamma=0.95
        if gama == 0.95:
            fig = plot_descent_path(module=L1, descent_path=weights_list[: t_fin[0]+1])
            fig.update_layout(title= "exponential decay of L1 with gama={0}".format(gama))
            fig.show()

        fig_convergence.add_trace(go.Scatter(x=np.arange(t_fin[0]+1), y=list_vals[:t_fin[0]+1], mode="lines",
                                 name="eta = " + str(gama)))
    fig_convergence.update_layout(title="exponential descent with L1 norm over increasing iterations".format("L1"),
                                legend_title = "Chosen eta")
    fig_convergence.show()



def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    from IMLearn.metrics.loss_functions import misclassification_error
    from IMLearn.model_selection import cross_validate

    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    import  sklearn.metrics as metrics

    logistic_regressor = LogisticRegression(solver=GradientDescent( learning_rate=FixedLR(1e-4), max_iter = 20000))
    logistic_regressor.fit(X_train,y_train)

    y_pred = logistic_regressor.predict_proba(X_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred , drop_intermediate=True)
    fig = go.Figure()
    fig.layout.title = f"ROC Curve"
    fig.add_trace(go.Scatter(x=fpr, y=tpr))
    fig.show()

    # ultimate alpha
    alpha_best = thresholds[np.argmax((tpr - fpr))]
    print("the chosen value for alpha star is "+ str(round(alpha_best,2)))

    # regressing on it:
    log_alpha_star = LogisticRegression(alpha=alpha_best)
    log_alpha_star.fit(X_train, y_train)
    ultimate_alpha_loss = round(log_alpha_star.loss(X_test, y_test),3)
    print("the loss for ultimate alpha is:" + str(ultimate_alpha_loss))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values of regularization parameter
    for penalty_func in ["l1", "l2"]:
        lambdas = [ 0.002, 0.02, 0.005, 0.01, 0.05, 0.1]
        all_validation_scores = []
        for lam in lambdas:
            print(lam)
            logistic_reg = LogisticRegression(lam=lam, penalty=penalty_func, alpha=0.5,
                                              solver=GradientDescent(learning_rate=FixedLR(1e-4),   max_iter=20000))

            train_score, validation_score = cross_validate(logistic_reg, X_train, y_train, misclassification_error)
            all_validation_scores.append(validation_score)

        # check which lamda is best according to lowest val error:
        lambda_best_model = lambdas[np.argmin(all_validation_scores)]

        print(f"best lambda value for {penalty_func} is " + str(lambda_best_model))

        fitted_module = LogisticRegression(lam=lambda_best_model, penalty=penalty_func, alpha=0.5,
                                 solver=GradientDescent(
                                     learning_rate=FixedLR(1e-4),
                                     max_iter=20000))
        fitted_module.fit(X_train, y_train)
        print(f"model error on this lambda is "  f"{fitted_module.loss(X_test, y_test)}")

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
