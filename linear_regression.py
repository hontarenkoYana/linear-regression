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


def draw(X, y, learned_coef_mse, learned_coef_mae, sklearn_coef):
    """
    Function to draw results.
    :param X: feature vector
    :param y: target value
    :param learned_coef_mse: learned by custom function with mse loss coefficient
    :param learned_coef_mae: learned by sklearn function with mae coefficient
    :param sklearn_coef: learned by sklearn function with mae coefficient
    """
    print(f"Learned by custom function learned with mse loss coefficient: {learned_coef_mse[1, 0]}, bias: {learned_coef_mse[0, 0]}")
    print(f"Learned by custom function learned with mae loss coefficient: {learned_coef_mae[1, 0]}, bias: {learned_coef_mae[0, 0]}")
    print(f"Learned by sklearn function coefficient: {sklearn_coef[1, 0]}, bias: {sklearn_coef[0, 0]}")
    plt.scatter(X, y, color='black')
    plt.legend(['data'])
    plt.plot(X, X * learned_coef_mse[1, 0] + learned_coef_mse[0, 0], color="blue")
    plt.legend(['My linear regression with mse'])
    plt.plot(X, X * learned_coef_mae[1, 0] + learned_coef_mae[0, 0], color="green")
    plt.legend(['My linear regression with mae'])
    plt.plot(X, X * sklearn_coef[1, 0] + sklearn_coef[0, 0], color="red")
    plt.legend(['Sklearn linear regression'])
    plt.show()


def linear_regression(X, theta):
    """
    Linear regression function.
    :param X: feature vector
    :param theta: predicted coefficient
    :return: X * theta
    """
    return X * theta[1] + theta[0]


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


def mae(predicted, true):
    """
    :param predicted: predicted values
    :param true: target values
    :return: mean absolute error between true and predicted values
    """
    return 1/(2 * predicted.shape[0]) * np.sum(np.abs(predicted - true))


def mae_derivative(predicted, true, X):
    """
    :param predicted: predicted values
    :param true: target values
    :param X: feature vector
    :return: derivative of mean squared error between true and predicted values
    """
    greater = predicted > true
    greater = np.where(greater, 1, -1)
    return np.sum(greater)


def gradient_descent(theta, X, y, epsilon, precision, M, loss="mse"):
    """
    Optimization of loss mse function for linear regression with gradient descend algorithm.
    :param theta: coefficients of linear regression
    :param X: feature vector
    :param y: target values vector
    :param epsilon: coefficient to interrupt cycle
    :param precision: precision of predicted values
    :param M: maximum number of iteration
    :param loss: loss function
    :return: new coefficients of linear regression
    """
    k = 0
    theta_loss_df = theta.copy()

    if loss == "mse":
        derivative = mse_derivative
    else:
        derivative = mae_derivative
    prediction = linear_regression(X, theta)
    theta_loss_df[0] = -derivative(prediction, y, 1)
    theta_loss_df[1] = -derivative(prediction, y, X)
    l_r = 0.5
    theta_new = theta + (l_r * theta_loss_df)
    while norm(theta_loss_df) > epsilon and k < M:
        prediction = linear_regression(X, theta)
        prediction_new = linear_regression(X, theta_new)
        mse_f = mse(prediction, y)
        mse_new_f = mse(prediction_new, y)
        if (mse_new_f - mse_f) > 0:
            l_r = l_r / 2
        else:
            if norm(theta_new - theta) > precision or np.abs(mse_new_f - mse_f) > precision:
                k += 1
                theta = theta_new.copy()
                prediction = linear_regression(X, theta)
                theta_loss_df[0] = -derivative(prediction, y, 1)
                theta_loss_df[1] = -derivative(prediction, y, X)
            else:
                k = M + 1
        theta_new = theta + (l_r * theta_loss_df)
    return theta_new


X, y = data()
learned_coef_mse = gradient_descent(np.array([1.5, 1.5]).reshape(-1, 1), X, y, 0.001, 0.001, 1000, "mse")
learned_coef_mae = gradient_descent(np.array([1.5, 1.5]).reshape(-1, 1), X, y, 0.001, 0.001, 1000, "mae")
lr = LinearRegression()
lr.fit(X, y)
sklearn_coef = np.array([lr.intercept_, lr.coef_])
draw(X, y, learned_coef_mse, learned_coef_mae, sklearn_coef)
