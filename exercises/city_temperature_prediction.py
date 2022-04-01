import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    import sys
    sys.path.append("../")
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    sys_str = 'C://Users//user//Documents//Uni/year B/IML/IML.HUJI/datasets/City_Temperature.csv'
    df = load_data(sys_str)
    # Question 2 - Exploring data for specific country
    israel_dataset = df[df['Country'] == 'Israel']
    israel_dataset.sort_values(["DayOfYear"])
    israel_dataset["Year_Str"] = israel_dataset["Year"].astype(str)

    fig1 = px.scatter(israel_dataset, x='DayOfYear', y='Temp', color='Year_Str')
    fig1.update_yaxes(range=[0, 30])
    fig1.show()
    # what polynomial degree might be suitable for the data
    for i in range(0, 2):
        mymodel = np.poly1d(np.polyfit(israel_dataset['DayOfYear'], israel_dataset['Temp'], 3))
        myline = np.linspace(1, 365, 1)
        fig1.plot(myline, mymodel(myline))
        fig1.show()

#    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
#    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
 #   raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
  #  raise NotImplementedError()