from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from statistics import mean, stdev
pio.templates.default = "simple_white"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    clean = pd.read_csv(filename)

    # logical contradictions to ignore:
    clean = clean[(clean.bedrooms >= 0) & (clean.price >= 0) & (clean.bathrooms >= 0)]
    clean = clean[(clean.sqft_lot15 >= 0)]

    # treating outliers:
    clean = clean[clean["bedrooms"] < 25]
    clean = clean[clean["price"] < 6000000]
    clean = clean[clean["sqft_lot"] < 800000]
    clean = clean[clean["sqft_living"] < 10000]
    clean = clean[clean["sqft_basement"] < 4000]

    # data which is not correlated with the price
    clean = clean.drop(['long', 'lat', 'date', 'id'], 1)

    # handling categorical data
    clean['floors'] = clean['floors'].astype(int)
    clean = pd.get_dummies(clean, columns=['zipcode'])

    # treating na
    clean = clean.dropna()

    #dataset_info(clean)

    # check correlation
    # corr = clean.corrwith(clean['price'])
    # corr = corr.sort_values()
    # corr = corr[(corr > 0.1)]
    # print(corr)
    # list_cols = corr.axes
    # not_ren_tbl = clean[list_cols[0]]

    return clean.drop("price", 1) , clean.price


def dataset_info(df):
    print("get shape")
    print(df.shape)
    print("get col-names")
    print(df.columns)
    print("get description")
    print(df.describe())
    print("get count")
    print(df.count())
    print("get minimal value")
    print(df.min())
    print("get maximal value")
    print(df.max())



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        # calculate pearson correlation between variables:
        denominator = np.std(X[feature]) * np.std(y)
        pearson = np.cov(X[feature], y)[0][1]/denominator

        # plot the relevant chart
        fig = go.Figure([go.Scatter(x=X[feature], y=y, showlegend=True, mode='markers')],
                    layout=go.Layout(title=f"Correlation Between {feature} and response<br>"
                                               f"<sup>Pearson Correlation {pearson}</sup>"))
        fig.update_layout(xaxis_title=f"{feature}", yaxis_title="response")
        fig.write_image(output_path + r"\price_and_" + feature + ".png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    import sys

    sys.path.append("..//")
    sys_str = 'C://Users//user//Documents//Uni/year B/IML/IML.HUJI/datasets/house_prices.csv'
    df, price_vector = load_data(sys_str)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, price_vector, r"C:\Users\user\Documents\Uni\year B\IML\charts")

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(df, price_vector, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    regression = LinearRegression(False)
    avg_loss = []
    std_loss = []
    lower_bound_CI = []
    upper_bound_CI = []

    percentage = np.linspace(10, 100, num=90)

    for p in percentage:
        loss_for_p = []
        for time_sampled in range(1, 10, 1):

            # get a % of the training data
            df_train = X_train.sample(frac=p/100, random_state=time_sampled)
            intercept_train = y_train.sample(frac=p/100, random_state=time_sampled)

            # fit the model using the portion sampled
            regression.fit(df_train.to_numpy(), intercept_train.to_numpy().flatten())
            loss_for_p.append(regression.loss(X_test.to_numpy(),
                                              y_test.to_numpy().flatten()))
        # check for mean and std for the % :
        avg_loss.append(mean(loss_for_p))
        std_loss.append(stdev(loss_for_p))

        # [mean - 2*std , [mean + 2*std ] for the percentage
        lower_bound_CI.append(mean(loss_for_p) - 2*stdev(loss_for_p))
        upper_bound_CI.append(mean(loss_for_p) + 2*stdev(loss_for_p))



    avg_loss = np.array(avg_loss)
    std_loss = np.array(std_loss)

    # the chart is just like the one given as an example in the book
    fig4 = go.Figure([go.Scatter(x=percentage, y=avg_loss,
                                 mode="markers+lines",
                                 name="avg Loss",
                                 line=dict(dash="dash"),
                                 marker=dict(color="green")),
                      go.Scatter(x=percentage,
                                 y=(lower_bound_CI),
                                 fill='tonexty',
                                 mode="lines",
                                 name="Lower bound CI",
                                 line=dict(color="lightgrey"),
                                 showlegend=False),
                      go.Scatter(x=percentage,
                                 y=(upper_bound_CI),
                                 fill='tonexty',
                                 name="Upper bound CI",
                                 mode="lines",
                                 line=dict(color="lightgrey"),
                                 showlegend=False)])
    fig4.update_layout(title_text="mean and std of the loss function <br> over a growing percentage of samples")

    fig4.show()


