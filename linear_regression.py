import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from scipy.linalg import norm
from sklearn.preprocessing import StandardScaler


def data():
    """
    Function to load data.
    :return: X - feature vector and y - target vector.
    """
    scaler = StandardScaler()
    # Load data
    boston_data_filename = load_boston()["filename"]
    data = pd.read_csv(boston_data_filename, header=1)
    data['target'] = load_boston()["target"]
    X = data["AGE"].values.reshape(-1, 1)
    y = data['target'].values.reshape(-1, 1)
    #Scale data
    X = scaler.fit_transform(X)
    return X, y


def draw(X, y, learned_coef, sklearn_coef):
    """
    Function to draw results.
    :param X: feature vector
    :param y: target value
    :param learned_coef: learned by custom function coefficient
    :param sklearn_coef: learned by sklearn function coefficient
    """
    print(f"Learned by custom function coefficient: {learned_coef[1, 0]}, bias: {learned_coef[0, 0]}")
    print(f"Learned by sklearn function coefficient: {sklearn_coef[1, 0]}, bias: {sklearn_coef[0, 0]}")
    plt.scatter(X, y, color='black')
    plt.legend(['data'])
    plt.plot(X, X * learned_coef[1, 0] + learned_coef[0, 0], color="blue")
    plt.legend(['My linear regression'])
    plt.plot(X, X * sklearn_coef[1, 0] + sklearn_coef[0, 0], color="red")
    plt.legend(['Sklearn linear regression'])
    plt.show()


def linear_regression(X, theta0, theta1):
    """
    Linear regression function.
    :param X: feature vector
    :param theta: predicted coefficient
    :return: X * theta
    """
    return X * theta1 + theta0


def mse(predicted, true):
    """
    :param predicted: predicted values
    :param true: target values
    :return: mean squared error between true and predicted values
    """
    return 1/(2 * predicted.shape[0]) * np.sum((predicted - true)**2)


def mse_derivative(predicted, true, X):
    """
    :param predicted: predicted values
    :param true: target values
    :param X: feature vector
    :return: derivative of mean squared error between true and predicted values
    """
    return (2/(predicted.shape[0])) * np.sum((predicted - true) * X)


def gradient_descent(theta0, theta1, precision, l_r, X, y):
    """
    Optimization of loss mse function for linear regression with gradient descend algorithm.
    :param theta0: bias variable (zero coefficient of linear regression)
    :param theta1: coefficient of linear regression
    :param precision: precision of predicted values
    :param l_r: learning rate - step size at each iteration
    :param X: feature vector
    :param y: target values vector
    :return: new coefficients of linear regression
    """
    prediction = linear_regression(X, theta0, theta1)
    theta0_loss_df = - mse_derivative(prediction, y, 1)
    theta1_loss_df = - mse_derivative(prediction, y, X)
    theta0_new = theta0 + (l_r * theta0_loss_df)
    theta1_new = theta1 + (l_r * theta1_loss_df)
    while norm(theta1_new - theta1) > precision:
        theta0 = theta0_new
        theta1 = theta1_new
        prediction = linear_regression(X, theta0, theta1)
        theta0_loss_df = - mse_derivative(prediction, y, 1)
        theta1_loss_df = - mse_derivative(prediction, y, X)
        theta0_new = theta0 + (l_r * theta0_loss_df)
        theta1_new = theta1 + (l_r * theta1_loss_df)

    return np.array([theta0_new, theta1_new]).reshape(-1, 1)


X, y = data()
learned_coef = gradient_descent(np.array([1.5]).reshape(-1, 1), np.array([1.5]).reshape(-1, 1), 0.001, 0.05, X, y)
lr = LinearRegression()
lr.fit(X, y)
sklearn_coef = np.array([lr.intercept_, lr.coef_])
draw(X, y, learned_coef, sklearn_coef)
