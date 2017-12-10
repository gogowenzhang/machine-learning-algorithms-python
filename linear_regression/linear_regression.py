import numpy as np

def predict(X, coeffs):
    return np.dot(X, coeffs)

def cost(X, y, coeffs, has_intercept=False):
    residuals = predict(X, coeffs) - y
    return np.dot(residuals, residuals) 

def gradient(X, y, coeffs, has_intercept=False):
    y_predict = predict(X, coeffs)
    return np.dot(X.T, y_predict-y)



