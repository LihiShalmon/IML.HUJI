from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
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
    clean = clean[clean["sqft_lot"] < 1000000]


    # data which is not correlated with the price
    clean = clean.drop(['long', 'lat', 'date', 'id'], 1)

    # handling categorical data
    clean["zipcode"] = clean["zipcode"].astype(int)

    # treating na
    clean = clean.dropna()

    # renovated quantiles
    clean_ren_recently = clean[(clean.yr_renovated > 0)]
    clean_not_ren = clean[(clean.yr_renovated == 0)]

    #dataset_info(clean)
    # check
    print("all data@@@@")
    corr = clean.corrwith(clean['price'])
    corr = corr.sort_values()
    corr = corr[(corr > 0.3)]
    print(corr)
    list_cols = corr.axes
    not_ren_tbl = clean[list_cols[0]]

    clean.insert(0, 'intercept', 1, True)
    return clean.drop("price", 1) , clean.price


def dataset_info(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            df.hist(column=col)
        except ValueError:
            print('This column can not be represented as a histogram')

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
    print("head")
    print(X.head())
    # for f in X:
    #     rho = np.cov(X[f], y)[0,1] / (np.std(X[f]) * np.std(y))
    #     print(rho)
    for feature in X:
        # calculate pearson correlation between variables:
        denominator = np.std(X[feature]) * np.std(y)
        cov = np.cov(X[feature], y)[0, 1]/denominator

        pearson = cov / denominator
        fig = go.Figure([go.Scatter(x=X[feature], y=y, showlegend=True,
                                marker=dict(color="black", opacity=.7),
                                line=dict(color="black", dash="dash", width=1))],
                    layout=go.Layout(title=f"Correlation Between {feature} and Response<br>"
                                               f"<sup>Pearson Correlation {pearson}</sup>",
                                     labels={"x": f"{feature}", "y": "Response "}))

        fig.write_image("pearson_correlation.%s.png" % feature)
    return "path"


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    import sys

    sys.path.append("..//")
    sys_str = 'C://Users//user//Documents//Uni/year B/IML/IML.HUJI/datasets/house_prices.csv'
    df, price_vector = load_data(sys_str)

    # Question 2 - Feature evaluation with respect to response

    # price_vector = df.loc[:, "price"]
    print(price_vector.mean)
    feature_evaluation(df, price_vector, "charts")

    # Question 3 - Split samples into training- and testing sets.

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    #raise NotImplementedError()
