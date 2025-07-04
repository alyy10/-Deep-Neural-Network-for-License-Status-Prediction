# Deep Neural Network for License Status Prediction

## Overview

This project focuses on building a Deep Neural Network (DNN) model to predict the status of business licenses based on various features. The dataset contains information about **86K different businesses** across multiple features, and the target variable is the **license status**, which has five categories (issued, renewed, canceled, pending, under review).

We will cover the process of building, training, evaluating, and deploying a deep learning model using **Python** and deep learning libraries such as **TensorFlow** and **h2o.ai**.

## Aim

The goal of this project is to introduce the concept of **Deep Neural Networks (DNN)** and demonstrate their implementation for a classification problem. You will learn how to:
- Prepare and clean data.
- Build a baseline model using h2o.
- Implement a DNN from scratch using TensorFlow.
- Deploy the model using **Flask** and **Gunicorn**.

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
    - **Pandas**: Data manipulation and analysis.
    - **Seaborn** & **Matplotlib**: Data visualization.
    - **Numpy**: Numerical computing.
    - **Scikit-learn**: Machine learning algorithms and data splitting.
    - **h2o.ai**: Baseline model building.
    - **TensorFlow**: Building and training the deep neural network.
    - **Flask**: API deployment.
    - **Gunicorn**: Production server for the Flask app.

## Project Structure

```plaintext
├── input/
│   ├── License_Data.csv
│   └── test_data.csv
├── output/
│   ├── columns.mapping
│   └── dnn-model
├── src/
│   ├── Engine.py
│   ├── ML_Pipeline/
│   │   ├── __init__.py
│   │   ├── Preprocess.py
│   │   ├── Predict.py
│   │   ├── Train_Model.py
│   │   └── Utils.py
│   ├── __init__.py
│   └── wsgi.py
├── requirements.txt
├── readme.md
└── Deep-Neural-Network.ipynb
