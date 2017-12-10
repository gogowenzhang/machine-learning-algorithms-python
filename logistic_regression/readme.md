# Implementation with gradient descent method. 

# Test with generated dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_classes=2,
                            random_state=0)

# In a terminal, you should be able to run your function like this:
import logistic_regression_functions as f
from GradientDescent import GradientDescent
gd = GradientDescent(f.cost, f.gradient, f.predict)
gd.fit(X, y)
print "coeffs:", gd.coeffs
predictions = gd.predict(X)


# Stochastic gradient descent
gd.fit_stochastic(X, y)
