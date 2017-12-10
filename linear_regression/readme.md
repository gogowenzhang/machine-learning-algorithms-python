# Implementation with gradient descent method. 

# Test with generated dataset
from sklearn.datasets import make_regression
X, y, coef = make_regression(n_features=10, coef=True)

# In a terminal, you should be able to run your function like this:
import linear_regression as f
from GradientDescent import GradientDescent
gd = GradientDescent(f.cost, f.gradient, f.predict)
gd.fit(X, y)
print "coeffs:", gd.coeffs
predictions = gd.predict(X)

# Compare coeffs and coef