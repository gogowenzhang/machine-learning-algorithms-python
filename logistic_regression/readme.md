# Logistic Regression

Implemented Logistic Regression using gradient descent as optimization method.

### Loss Function
The probability function: ![hypothesis](https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/logistic_regression/img/hypothesis.png)

The likelihood function: ![likehood](https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/logistic_regression/img/likelihood.png)

The loss function: ![loss](https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/logistic_regression/img/cost.png)

### Gradient Descent
Gradient of the loss function: ![gradient](https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/logistic_regression/img/gradient.png)

Each partial derivative: ![partial](https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/logistic_regression/img/partial.png)


### Test with generated dataset
```
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_classes=2,
                            random_state=0)
```

### In a terminal, you should be able to run your function like this:
```import logistic_regression_functions as f
from GradientDescent import GradientDescent
gd = GradientDescent(f.cost, f.gradient, f.predict)
gd.fit(X, y)
print "coeffs:", gd.coeffs
predictions = gd.predict(X)
```

### Stochastic gradient descent
```
gd.fit_stochastic(X, y)
```


### Package Dependencies
numpy==1.13.1

sklearn==0.18.1
