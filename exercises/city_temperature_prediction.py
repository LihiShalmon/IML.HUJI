import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def dataset_info(df):
    print("get types")
    print(df.dtypes)
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

    # clean anomalies in temperature
    df = df[df['Temp'] > -30]

    # get the date of year feature
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    sys_str = 'C://Users//user//Documents//Uni/year B/IML/IML.HUJI/datasets/City_Temperature.csv'
    df = load_data(sys_str)

    # Question 2 - Exploring data for specific country
    df_israel = df[df['Country'] == 'Israel']
    df_israel.sort_values(["DayOfYear"])
    df_israel["Year_Str"] = df_israel["Year"].astype(str)

    fig1 = px.scatter(df_israel, x='DayOfYear', y='Temp', color='Year_Str',
                      title='Temperature by the days of the year')
    fig1.update_yaxes(range=[0, 35])
    fig1.show()

    # what polynomial degree might be suitable for the data
    response = df_israel['Temp']
    X = df_israel['DayOfYear']
    # polyfit = PolynomialFitting(1)
    # polyfit._fit(X, response)
    # print(polyfit.model.coefs_)

    # question 2 B:
    temp_by_month = df_israel.groupby("Month").agg(stand_deviation=('Temp', 'std'))
    fig2 = px.bar(temp_by_month, y='stand_deviation',
                  title='Standard deviation of temperature by months')
    fig2.show()

    # Question 3 - Exploring differences between countries
    grouped_data = df.groupby(['Month', 'Country']).agg(stand_deviation=('Temp', 'std'),
                                                        mean_temp=('Temp', 'mean')).reset_index()


    print(grouped_data.head())
    fig3 = px.scatter(grouped_data, x='Month', y='mean_temp', color='Country',
                      error_y= 'stand_deviation',
                      title='The average monthly temperature and the standard deviations of each country')
    fig3.update_traces(mode='markers+lines')
    fig3.show()

    fig3.update_yaxes(range=[0, 35])
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    df_israel_no_response = df_israel.drop(['Temp','Year_Str','Date'], axis=1)
    israel_df_response = df_israel['Temp']
    X_train, y_train, X_test, y_test = split_train_test(df_israel_no_response['DayOfYear'],
                                                        israel_df_response, 0.75)
    loss_by_k = []
    for k in range(1,11,1):
        polinomial_fit = PolynomialFitting(k)
        polinomial_fit._fit(X_train, y_train)
        loss = polinomial_fit._loss(X_test.values, y_test.values)
        round_loss = round(loss,2)
        loss_by_k.append(round_loss)

    print(loss_by_k)
    fig4 = px.bar(loss_by_k, x=range(1,11,1), y=loss_by_k ,
                  title=('the MSE polinomial regression of degree K'))
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    # degree k=5 provides optimal prediction
    polinomial_fit = PolynomialFitting(5)
    polinomial_fit._fit(X_train, y_train)
    print(df.columns)
    country_loss = []

    unique_country = df['Country'].unique()
    for country in unique_country:
        df_prediction_country = df[df['Country'] == country]
        X_train, y_train, X_test, y_test = split_train_test(df_prediction_country['DayOfYear'],
                                                            df_prediction_country['Temp'], 0.75)

        loss = polinomial_fit._loss(X_test, y_test)
        country_loss.append(loss)

    print(country_loss)
    fig5 = px.bar(country_loss ,
                  x=unique_country,
                  y= country_loss,
                  title=('Loss by different countries'))
    fig5.show()

