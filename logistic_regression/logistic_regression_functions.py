import numpy as np


def predict_proba(X, coeffs):
    """Calculate the predicted conditional probabilities (floats between 0 and
    1) for the given data with the given coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.

    Returns
    -------
    predicted_probabilities: The conditional probabilities from the logistic
        hypothosis function given the data and coefficients.

    """
    return 1./(1+np.exp(-np.dot(X, coeffs.T)))


def predict(X, coeffs, thresh=0.5):
    """
    Calculate the predicted class values (0 or 1) for the given data with the
    given coefficients by comparing the predicted probabilities to a given
    threshold.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X and coeffs must align.
    thresh: Threshold for comparison of probabilities.

    Returns
    -------
    predicted_class: The predicted class.
    """
    predict = predict_proba(X, coeffs)
    return predict >= thresh

def cost(X, y, coeffs, lam=0.0, has_intercept=True):
    """
    Calculate the logistic cost function of the data with the given
    coefficients.

    Parameters
    ----------
    X: A 2 dimensional numpy array.  The data (independent variables) to use
        for prediction.
    y: A 1 dimensional numpy array.  The actual class values of the response.
        Must be encoded as 0's and 1's.  Also, must align properly with X and
        coeffs.
    coeffs: A 1 dimensional numpy array, the hypothesised coefficients.  Note
        that the shape of X, y, and coeffs must align.

    Returns
    -------
    logistic_cost: The computed logistic cost.
    """
    pp = predict_proba(X, coeffs)
    ridge_penalty = np.dot(coeffs, coeffs)
    if has_intercept:
        ridge_penalty -= coeffs[0] * coeffs[0]
    return - np.dot(y, np.log(pp)) - np.dot((1-y), np.log(1-pp)) + lam * ridge_penalty

def cost_one_datapoint(x, y, coeffs):
    """
    Calculate the logistic cost function of a single data point using the given
    coefficients.
    Parameters
    ----------
    x: ndarray, shape (n_features, )
        The data (independent variables) to use for prediction.
    y: Integer, 0 or 1 
        The actual class values of the response.
    coeffs: ndarray, shape (n_features)
        The hypothosized coefficients of the logistic regression.
    Returns
    -------
    logistic_cost: float
        The computed logistic cost.
    """
    linear_pred = np.sum(x * coeffs)
    p = 1 / (1 + np.exp(-linear_pred))
    return (- y * np.log(p) - (1 - y) * np.log(1 - p))


def gradient(X, y, coeffs, lam=0.0, has_intercept=True):
    p = predict_proba(X, coeffs)
    ridge_grad = 2 * coeffs
    if has_intercept:
        ridge_grad[0] = 0.0
    return np.dot(X.T, p-y) + lam * ridge_grad


def gradient_one_datapoint(x, y, coeffs):
    """
    Calculate the gradient of the logistic cost function evaluated at a single
    data point.
    Parameters
    ----------
    x: ndarray, shape (n_features, )
        The data (independent variables) to use for prediction.
    y: Integer, 0 or 1 
        The actual class values of the response.
    coeffs: ndarray, shape (n_features)
        The hypothosized coefficients of the logistic regression.
    Returns
    -------
    logistic_grad: ndarray, shape (n_features, )
        The computed gradient of the logistic cost.
    """
    linear_pred = np.sum(x * coeffs)
    p = 1 / (1 + np.exp(-linear_pred))
    return (x * (p - y))

