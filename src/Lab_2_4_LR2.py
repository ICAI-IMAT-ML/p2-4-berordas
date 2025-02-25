import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session

        # Store the intercept and the coefficients of the model
        if X.shape[0] != len(y):
            raise ValueError("X e y deben tener las mismas dimensiones.")

        X_augmented = np.c_[np.ones(X.shape[0]), X]  
        
        beta = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y

        self.intercept = beta[0] 
        self.coefficients = beta[1:] 

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01

        # Implement gradient descent 
        for epoch in range(iterations):
            predictions = np.dot(X[:, 1:], self.coefficients) + self.intercept
            error = predictions - y

            # TODO: Write the gradient values and the updates for the paramenters
            gradient = np.dot(X[:, 1:].T, error) / m
            intercept_gradient = np.sum(error) / m

            self.intercept -= learning_rate * intercept_gradient
            self.coefficients -= learning_rate * gradient

            # Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # Predict when X is only one variable
            predictions = X * self.coefficients + self.intercept
        else:
            # Predict when X is more than one variable
            predictions = self.intercept + np.dot(X, self.coefficients)

        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    if len(y_true) == len(y_pred):
        # MSE
        # Calculate MSE
        mse = 0
        for i in range(len(y_true)):
            mse += (y_true[i] - y_pred[i]) ** 2

        mse = (mse / len(y_true))

        # R^2 Score
        # Calculate R^2
        ss_res = 0
        ss_tot = 0
        media_y_true = np.mean(y_true)

        for i in range(len(y_true)):
            ss_res += (y_true[i] - y_pred[i]) ** 2
            ss_tot += (y_true[i] - media_y_true) ** 2

        r_squared = 1 - (ss_res / ss_tot)

        # Root Mean Squared Error
        # Calculate RMSE
        rmse = mse ** (1 / 2)

        # Mean Absolute Error
        # Calculate MAE
        mae = 0
        for i in range(len(y_true)):
            mae += abs(y_true[i] - y_pred[i])

        mae = mae / len(y_true)

    else:
        r_squared = None
        rmse = None
        mae = None

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    for index in sorted(categorical_indices, reverse=True):
        # Extract the categorical column
        categorical_column = X_transformed[:, index]

        # Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([unique_values == val for val in categorical_column], dtype=int)

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.insert(X_transformed, index, one_hot.T, axis=1)

    return X_transformed
