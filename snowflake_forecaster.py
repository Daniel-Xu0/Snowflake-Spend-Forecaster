"""
Daniel Xu
December 18th, 2022
Drift, Inc.
Description:
    This script is my attempt at building a linear regression and random forest regressor that
    can accurately predict the company's Snowflake spend given past spend data over the past two years.
Note to self:
        It might be best to take out all the warehouses that aren't being used anymore.
        They might be screwing up our current credit usage predictions bc they're no longer relevant and have
        no predictive power over current Snowflake credit usage
"""

import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def timer_func(func):
    """ Timer decorator - shows execution time of function passed

    :param func: Function
    :return: Execution time in seconds
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Your model {func.__name__!r} finished in {(t2-t1):.4f}s')
        return result
    return wrap_func


class SnowflakeForecaster:
    def __init__(self, snowflake_csv):
        """
        Constructor method for Snowflake Forecaster class
        :param csv (str): name of csv file we're reading in
        :param snowflake_csv (df): csv file converted into a Pandas dataframe
        :param timeframe (int): timeframe we want to predict the cost for
        """
        self.snowflake_usage = self.transform_snowflake_data(snowflake_csv)

    @staticmethod
    def transform_snowflake_data(snowflake_csv):
        """
        Run "select * from snowflake.account_usage.warehouse_metering_history
            where start_time > dateadd(year, -2, current_date());"
            in Snowflake to get the necessary data to run this model
        :param snowflake_csv: csv file downloaded from Snowflake query
        :return: Snowflake usage data with newly created feature columns for machine learning models
        """
        snowflake_usage = pd.read_csv(snowflake_csv)

        # Getting rid of columns we don't need
        snowflake_usage = snowflake_usage.loc[:, ['START_TIME', 'WAREHOUSE_ID', 'WAREHOUSE_NAME', 'CREDITS_USED']]

        # Grouping usage data by the day
        snowflake_usage['START_TIME'] = snowflake_usage['START_TIME'].str[:10]
        snowflake_usage = snowflake_usage.groupby(['START_TIME', 'WAREHOUSE_ID', 'WAREHOUSE_NAME']) \
            ['CREDITS_USED'].sum().reset_index()

        # Indexing start time data to better group our data
        snowflake_usage['YEAR'] = snowflake_usage['START_TIME'].str[:4].astype(int)
        snowflake_usage['MONTH'] = snowflake_usage['START_TIME'].str[5:7].astype(int)
        snowflake_usage['DAY'] = snowflake_usage['START_TIME'].str[8:].astype(int)
        snowflake_usage['DAY_OF_WEEK'] = pd.to_datetime(snowflake_usage['START_TIME'], format='%Y-%m-%d').dt.day_name()

        # Dummy columns for all warehouse IDs and day_of_week as well - one-hot-encoding
        snowflake_usage = pd.concat([pd.get_dummies(snowflake_usage, prefix='BOOLEAN',
                                                    columns=['WAREHOUSE_NAME', 'DAY_OF_WEEK']), snowflake_usage],
                                    axis=1).T.drop_duplicates().T

        return snowflake_usage

    @staticmethod
    def plot_usage(df, x, hover_data, y='CREDITS_USED', color='WAREHOUSE_NAME', symbol=None, forecast=True):
        """
        Simple plotly scatter plot of Snowflake usage data

        :param forecast:
        :param df: dataframe to plot data of
        :param x: x-axis
        :param y: y-axis; default is number of credits
        :param color: which column will determine color of points (appears in legend); default is warehouse
        :param hover_data: what information will appear when hovering over points
        :param symbol: what defines the symbol each point shows up as
        :param forecast: whether forecasted data is included - affects title
        :return:
        """
        # Create plot title
        title = 'Historical + Predicted Snowflake Credit Usage' if forecast else 'Snowflake Credit Usage'

        # Create plotly scatter plot of credit usage
        fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover_data, symbol=symbol, title=title)
        fig.show()

    @staticmethod
    def randomized_search_cv():
        """ Create a random grid of parameters for the random forest regressor to try

        :return: random grid of params
        """
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        return random_grid

    def build_train_test_data(self, x_feat_list, train_size, random_state):
        """ Build the training and testing data tailored towards our Snowflake credit usage df

        :param x_feat_list:
        :param train_size:
        :param random_state:
        :return: x_features_train, y_features_train, x_features_test, y_features_test
        """
        # What features we want to incorporate into our model and what we want to predict
        y_feat = 'CREDITS_USED'

        # Get dependent and independent variables
        x = self.snowflake_usage.loc[:, x_feat_list].values
        y = self.snowflake_usage.loc[:, y_feat].values

        # split dataset in train and testing set
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

        return x_train, x_test, y_train, y_test

    def build_forecast_data(self, predict_range):
        """ Create new sample data for the next month that we can use our models to predict on

        :param predict_range: # of days of data we want to forecast and visualize
        :return: dataframe including both historical and forecasted credit usage
        """
        # Retrieve final data within usage data
        last_date = self.snowflake_usage.iloc[-1, 0]

        # Turning date into date object
        last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
        next_days = [(last_date + timedelta(days=i)).isoformat() for i in range(predict_range)]

        # Get list of unique warehouses and ids
        warehouse_names = list(self.snowflake_usage['WAREHOUSE_NAME'].unique())
        warehouse_ids = list(self.snowflake_usage['WAREHOUSE_ID'].unique())

        # Generate new sample data by combining future dates with warehouses
        new_data = []
        for date in next_days:
            year = date[:4]
            month = date[5:7]
            day = date[8:]
            for i in range(len(warehouse_names)):
                new_data.append([date, warehouse_ids[i], warehouse_names[i], year, month, day])

        # Create dummy data dataframe to predict credit usage on
        dummy_df = pd.DataFrame(new_data,
                                columns=['START_TIME', 'WAREHOUSE_ID', 'WAREHOUSE_NAME', 'YEAR', 'MONTH', 'DAY'])
        dummy_df['DAY_OF_WEEK'] = pd.to_datetime(dummy_df['START_TIME'], format='%Y-%m-%d').dt.day_name()
        sample_df = pd.concat([pd.get_dummies(dummy_df, prefix='BOOLEAN',
                                              columns=['WAREHOUSE_NAME', 'DAY_OF_WEEK']), dummy_df],
                              axis=1).T.drop_duplicates().T
        return sample_df

    def visualize_model_forecast(self, regressor, x_feat_list, predict_range):
        """ Use Plotly to visualize historical and forecast data as one plot

        :param regressor: regression model to predict data with
        :param x_feat_list: list of X_features that we're used on the regressor model
        :param predict_range: # of days of data we want to forecast and visualize
        """
        # Build forecast data to predict credit usage on
        forecast_data = self.build_forecast_data(predict_range)

        # Use linear regression model to forecast X # days of credit usage
        forecast_variables = forecast_data.loc[:, x_feat_list].values
        forecast_data['CREDITS_USED'] = regressor.predict(forecast_variables)

        # Plot credit usage forecast over the next X # of days
        usage_df = pd.concat([self.snowflake_usage, forecast_data],
                             keys=['HISTORICAL', 'FORECASTED'],
                             names=['DATA_PERIOD', 'INDEX'])
        usage_df.reset_index(inplace=True)
        self.plot_usage(usage_df, 'START_TIME', ['START_TIME', 'CREDITS_USED'], symbol='DATA_PERIOD')

    def linear_regression_model(self, x_feat_list, train_size, random_state, verbose=True):
        """ Create a linear regression model based on Snowflake usage data

        Args:
            x_feat_list (list): list of all features in model
            train_size (float): value between 0.0 - 1.0; what the training size of our data will be
            random_state (int): what seed to give the model in order to get fixed results
            verbose (bool): toggles command line output.

        Returns:
            reg (LinearRegression): model fit to data.
        """
        # initialize regression object
        regressor = LinearRegression()

        # split dataset in train and testing set
        x_train, x_test, y_train, y_test = self.build_train_test_data(x_feat_list, train_size, random_state)

        # fit regression
        regressor.fit(x_train, y_train)

        # compute / store r2
        y_pred = regressor.predict(x_test)

        if verbose:
            # compute / print r2
            r2 = r2_score(y_true=y_test, y_pred=y_pred)
            print(f'R Squared score for Linear Regression: \n{r2:.3}')

        return regressor

    @timer_func
    def run_linear_regression(self, predict_range, train_size=.75, random_state=0, visualization=False):
        """ Use regression model function to predict next month's warehouse usage

        Args:
            predict_range (int): # of days of data we want to forecast and visualize
            train_size (float): value between 0.0 - 1.0; what the training size of our data will be
            random_state (int): what seed to give the model for specific results
            visualization (boolean): plots historical and sample future data if = True
        Returns:
            linear regression model along with r^2 score and a plot of the next predict_range days worth of data
        """
        # What factors we want to incorporate and what we want to predict; credit_usage
        x_feat_list = self.snowflake_usage.drop(
            ['CREDITS_USED', 'START_TIME', 'WAREHOUSE_ID', 'WAREHOUSE_NAME', 'DAY_OF_WEEK'], axis=1).columns

        # Create linear regression model
        regressor = self.linear_regression_model(x_feat_list, train_size, random_state)

        # Create visualization if wanted
        if visualization:
            self.visualize_model_forecast(regressor, x_feat_list, predict_range)

    @timer_func
    def run_rf_regression(self, predict_range, n_iter=60, k_folds=5, train_size=.75,
                          random_state=0, visualization=False):
        """ Build and run a Random Forest Regressor

        :param n_iter: # of parameter combinations to try out for each fold
        :param k_folds: # of folds to use in cross validation
        :param predict_range: # of days of data we want to forecast and visualize
        :param train_size: value between 0.0 - 1.0; what the training size of our data will be
        :param random_state: what seed to give the model for specific results
        :param visualization: plots historical and sample future data if = True
        :return: random forest regressor model along with accuracy score and a plot of the next
                 predict_range days worth of data
        """
        # Create random forest regressor object
        rf = RandomForestRegressor()

        # Create random grid of training features
        random_grid = self.randomized_search_cv()

        # Random search of parameters, using 3-fold cross validation - Finds best combination of params
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                       n_iter=n_iter, scoring='neg_mean_absolute_error',
                                       cv=k_folds, verbose=2, random_state=random_state, n_jobs=-1,
                                       return_train_score=True)

        # What factors we want to incorporate and what we want to predict; credit_usage
        x_feat_list = self.snowflake_usage.drop(
            ['CREDITS_USED', 'START_TIME', 'WAREHOUSE_ID', 'WAREHOUSE_NAME', 'DAY_OF_WEEK'], axis=1).columns

        # split dataset into training and testing set
        x_train, x_test, y_train, y_test = self.build_train_test_data(x_feat_list, train_size, random_state)

        # Fit the random search model
        rf_regressor = rf_random.fit(x_train, y_train)

        # Find the best version of our random forest regressor and determine its accuracy
        best_rf_regressor = rf_regressor.best_estimator_
        print("Best random forest regressor accuracy score: \n", best_rf_regressor.score(x_test, y_test))

        if visualization:
            self.visualize_model_forecast(best_rf_regressor, x_feat_list, predict_range)


if __name__ == '__main__':
    drift_snowflake_forecast = SnowflakeForecaster('snowflake_usage.csv')
    drift_snowflake_forecast.run_linear_regression(31, visualization=True)
    drift_snowflake_forecast.run_rf_regression(31, n_iter=120, k_folds=4, train_size=.75, visualization=True)
