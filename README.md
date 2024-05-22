# US-GDP-before-and-after-COVID-19


This repository contains the code and data used for the study "US GDP Forecasting in the Face of COVID-19: A Comparative Study of Advanced Machine Learning and Neural Network Techniques." The study aims to compare the performance of various Machine Learning (ML) and Deep Learning (DL) models in forecasting US GDP before and after the COVID-19 pandemic.
Table of Contents

- Introduction
- Dataset
- Models
- Results
- Usage
- References
- License

## Introduction

The COVID-19 pandemic has significantly impacted global economies, including the United States. Accurate GDP forecasting is crucial for policymakers, economists, and stakeholders. This study compares the performance of various ML and DL models in predicting US GDP before and after the COVID-19 pandemic, using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and others.

## Dataset

The dataset used in this study includes various economic indicators for the United States, spanning from Q1 1970 to Q3 2023. The dataset is split into two environments:

- Pre-COVID-19: Data from Q1 1970 to Q4 2019.
- Post-COVID-19: Data from Q1 1970 to Q3 2023.

The dataset is stored in a file named dataset.csv and is ";" separated.

## Models

The study evaluates several ML and DL models, including:
Machine Learning Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Gradient Boosting

## Deep Learning Models

- Recurrent Neural Networks (RNN)
- Long Short-Term Memory Networks (LSTM)

Each model is evaluated using both K-Fold Cross-Validation and Time Series Cross-Validation.

## Results

The study found that the COVID-19 pandemic significantly impacted the performance of GDP forecasting models. Pre-COVID-19 models generally performed better than post-COVID-19 models. Time Series splits often outperformed K-Fold splits in the post-COVID-19 environment, indicating that temporal relationships became more significant due to the pandemic.