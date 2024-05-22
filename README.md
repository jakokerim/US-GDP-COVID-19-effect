# **US-GDP-before-and-after-COVID-19**


This repository contains the code and data used for the study "US GDP Forecasting in the Face of COVID-19: A Comparative Study of Advanced Machine Learning and Neural Network Techniques." The study aims to compare the performance of various Machine Learning (ML) and Deep Learning (DL) models in forecasting US GDP before and after the COVID-19 pandemic.

## Table of Contents

Table of Contents

- Introduction
- Dataset
- Methodology
- Models
- Results
- Conclusion

## Introduction

The COVID-19 pandemic has significantly impacted global economies, including the United States. Accurate GDP forecasting is crucial for policymakers, economists, and stakeholders. This study compares the performance of various ML and DL models in predicting US GDP before and after the COVID-19 pandemic, using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Dataset

The dataset used in this study includes various economic indicators for the United States, spanning from Q1 1970 to Q3 2023. The dataset is split into two environments:

- Pre-COVID-19: Data from Q1 1970 to Q4 2019.
- Post-COVID-19: Data from Q1 1970 to Q3 2023.

The dataset is stored in a file named dataset.csv and is ";" separated.

Notebooks contain codes to run the right splits and save the resulting new dataset. The two others CSV files are a result of data cleaning. 

## Data Description and Preparation

The data includes economic indicators such as:

    US GDP (AR)
    US unemployment
    US 10-Year Treasury Bonds Yield
    Non-Farm employment
    US CPI
    US Nominal Broad Effective Exchange Rate
    US Personal Current Taxes (AR)
    US Government Consumption & Investment (AR)
    US Current Account Balance

The data preprocessing steps involve:

    Importing the dataset.
    Parsing the date format from "Q1 1970" to "YYYY-MM-DD".
    Setting the dataset bounds for pre- and post-COVID-19 environments.

## Methodology
Cross-Validation

Cross-validation (CV) is used to assess the performance and robustness of the models. This study employs both K-Fold Cross-Validation and Time Series Cross-Validation to evaluate the models.
Variable Selection

Variables are selected based on correlation visualization, Augmented Dickey-Fuller (ADF) test, and Variance Inflation Factor (VIF) test. First differencing is applied to handle non-stationarity.

## Models

The study evaluates several ML and DL models, including:
Machine Learning Models

    Linear Regression: Uses Ordinary Least Squares (OLS) to estimate the coefficients.
    Ridge Regression: Introduces a penalty term to handle multicollinearity.
    Lasso Regression: Uses L1 norm for regularization and variable selection.
    Polynomial Regression: Captures non-linear relationships by fitting a polynomial equation.
    Gradient Boosting: An ensemble method that builds models incrementally to correct errors.

## Deep Learning Models

    Recurrent Neural Networks (RNN): Designed to recognize patterns in sequences of data.
    Long Short-Term Memory Networks (LSTM): Overcomes the limitations of RNNs by capturing long-term dependencies.

**Each model is evaluated using both K-Fold Cross-Validation and Time Series Cross-Validation.** 

## Results

The study found that the COVID-19 pandemic significantly impacted the performance of GDP forecasting models. Pre-COVID-19 models generally performed better than post-COVID-19 models. Time Series splits often outperformed K-Fold splits in the post-COVID-19 environment, indicating that temporal relationships became more significant due to the pandemic.

## Dependencies

Ensure you have the following dependencies installed (not always needed):
    pip install --upgrade pandas
    pip install --upgrade numexpr bottleneck
    pip install statsmodels
    pip install seaborn
    pip install --upgrade scikit-learn
    import pandas as pd 
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    import statsmodels.api as sm
    import tensorflow as tf
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.api import add_constant
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import make_scorer, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.decomposition import PCA
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed, Dropout
