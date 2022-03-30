from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
#import matplotlib.pyplot as plt

pio.templates.default = "simple_white"
MEAN_OF_SAMPLES = 10
VAR_SAMPLES = 1
REPS_Q1 = 1000


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    observations_vector = \
        np.random.normal(loc=MEAN_OF_SAMPLES, scale=VAR_SAMPLES, size=REPS_Q1)
    univ = UnivariateGaussian()
    univ.fit(observations_vector)
    print("(" + str(univ.mu_) + " , " + str(univ.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    samples = np.linspace(10, 1000, 100).astype(int)
    distance_between_means = {}
    for n in range(10, 1000 + 1, 10):
        distance_between_means[n] = np.absolute(
            univ.mu_ - observations_vector[0:n].mean())
    go.Figure(go.Scatter(x=list(distance_between_means.keys()),
                         y=list(distance_between_means.values()),
                         mode='lines', name='Q2_Chart'),
              layout=go.Layout(
                  title='Absolute value between estimated and approximated expectation'
                  , xaxis_title='number of samples'
                  , yaxis_title='distance from aproximated expectation')).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    x_values = np.sort(observations_vector)
    y_values = univ.pdf(x_values)
    fig3 = go.Figure(go.Scatter(x=x_values, y=y_values, mode='markers+lines'),
                     layout=go.Layout(
                         title='Empirical PDF of fitted model'
                         , xaxis_title='sample values'
                         , yaxis_title='sample PDF'))
    fig3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    requested_mu = [0, 0, 4, 0]
    requested_cov = np.array([[1, 0.2, 0, 0.5],
                              [0.2, 2, 0, 0],
                              [0, 0, 1, 0],
                              [0.5, 0, 0, 1]])

    observations_vector = \
        np.random.multivariate_normal(requested_mu,
                                      requested_cov,
                                      REPS_Q1)
    multi_var = MultivariateGaussian()
    multi_var.fit(observations_vector)
    print(str(multi_var.mu_))
    print(str(multi_var.cov_))

    # Question 5 - Likelihood evaluation
    # f1, f3
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    arr_likelihood = []
    arr_likelihood = np.zeros((200, 200))

    max_f1 = 0
    max_f3 = 0
    for i in range(len(f1)):
        for j in range(len(f3)):
            new_mu = np.array([f1[i], 0, f3[j], 0])
            # calculate the likelihood function of new mu
            arr_likelihood[i, j] = MultivariateGaussian.log_likelihood(new_mu,
                                                                       requested_cov,
                                                                       observations_vector)
            # searching for the model with the maximum log-likelihood
            if arr_likelihood[i, j] > arr_likelihood[max_f1, max_f3]:
                max_f1, max_f3 = i, j

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=f3, y=f1, z=arr_likelihood))
    fig.update_layout(title="Log likelihood Heatmap",
                      xaxis_title="f3 values", yaxis_title="f1 values")
    fig.show()

    # Question 6 - Maximum likelihood
    f1_val = str(round(f1[max_f1], 3))
    f3_val = str(round(f3[max_f3], 3))
    print("f1={f1}, f3={f3}".format(f1=f1_val, f3=f3_val))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
