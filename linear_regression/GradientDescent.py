import numpy as np
from sklearn.preprocessing import StandardScaler


class GradientDescent(object):
    """Perform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """

    def __init__(self, cost, gradient, predict_func,
                 fit_intercept=False,
                 standardize=False,
                 alpha=0.01,
                 num_iterations=10000,
                 step_size=0.0001,
                 cost_func_parameters=None):
        """Initialize the instance attributes of a GradientDescent object.
        Parameters
        ----------
        cost: A function with signature (2d np.array, 1d np.array, 1d np.array)
          -> 1d np.array 
            The cost function to be minimized.
        gradient: A function with signature (2d np.array, 1d np.array, 1d np.array)
          -> 1d np.array 
            The gradient of the cost function.
        predict_func: A function of signature (2d np.array, 1d np.array) -> 1d
          np.array
            A function to make predictions after the optimizaiton has converged.
        alpha: float
            The learning rate.
        num_iterations: integer.
            Number of iterations to use in the descent.
        Returns
        -------
        self:
            The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.cost_history = []
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.standardizer = None
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.step_size = step_size
        if cost_func_parameters is None:
            cost_func_parameters = {}
        self.cost_func_parameters = cost_func_parameters

    def fit(self, X, y):
        """Run the gradient descent algorithm for num_iterations repititions.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The training data for the optimization.
        y: ndarray, shape (n_samples, ).
            The training response for the optimization.
        Returns
        -------
        self:
            The fit GradientDescent object.
        """
        X = self.prepare_data_for_fit(X)
        self.coeffs = np.zeros(X.shape[1])
        for i in xrange(self.num_iterations):
            grad = self.gradient(X, y, self.coeffs,
                                 has_intercept=self.fit_intercept,
                                 **self.cost_func_parameters)
            self.coeffs = self.coeffs - self.alpha * grad
            cost = self.cost(X, y, self.coeffs,
                             has_intercept=self.fit_intercept,
                             **self.cost_func_parameters)
            self.cost_history.append(cost)
            if i >= 2 and self.has_converged(cost, self.cost_history[-2]):
                break
        return self

    def fit_stochastic(self, X, y):
        """Run the stochastic gradient descent algorithm for num_iterations
        epochs (passes through the data).
        X: ndarray, shape (n_samples, n_features)
            The training data for the optimization.
        y: ndarray, shape (n_samples, ).
            The training response for the optimization.
        Returns
        -------
        self:
            The fit GradientDescent object.
        """
        X = self.prepare_data_for_fit(X)
        self.coeffs = np.zeros(X.shape[1])
        X, y = self.shuffle_data_and_response(X, y)
        for i in xrange(self.num_iterations):
            for j in xrange(X.shape[0]):
                grad = self.gradient(X[j, :], y[j], self.coeffs)
                self.coeffs = self.coeffs - self.alpha * grad
                cost = self.cost(X[j, :], y[j], self.coeffs)
                self.cost_history.append(cost)
            has_converged = self.has_converged_stochastic(X.shape[0])
            if (i >= 2 and has_converged):
                break
        return self

    def predict(self, X):
        """Call self.predict_func to return predictions.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            Predictor data to make predictions for.
        Returns
        -------
        preds: ndarray, shape (n_samples)
            Array of predictions.
        """
        if self.standardize:
            X = self.standardize_data_transform(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.predict_func(X, self.coeffs)

    def add_intercept(self, X):
        """Add an intercept column to a matrix X.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
        Returns
        -------
        X: ndarray, shape (n_samples, n_features + 1)
            The original matrix X, but with a constant columns of 1's appended.
        """
        ones = np.ones(X.shape[0]).reshape(-1, 1)
        return np.concatenate([ones, X], axis=1)

    def standardize_data_fit(self, X):
        """Transform the features in X to have zero mean and unit variance,
        and memorize the parameters needed to do so so that future data may
        be transformed with the same standardization.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
        Returns
        -------
        X: ndarray, shape (n_samples, n_features)
            The original matrix X, but with the columns standardized.
        """
        self.standardizer = StandardScaler()
        self.standardizer.fit(X)
        return self.standardizer.transform(X)

    def standardize_data_transform(self, X):
        """Transform the features in X to have zero mean and unit variance
        using memorized parameters.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
        Returns
        -------
        X: ndarray, shape (n_samples, n_features)
            The original matrix X, but with the columns standardized.
        """
        return self.standardizer.transform(X)

    def prepare_data_for_fit(self, X):
        """Standardize and add an intercept column to data if requeted.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
        Returns
        -------
        X: ndarray, shape (n_samples, n_features (+1)?)
            The original matrix X, but maybe with the columns standardized and
            maybe with an intercept column appended.
        """
        if self.standardize:
            X = self.standardize_data_fit(X)
        if self.fit_intercept:
            X = self.add_intercept(X)
        return X

    def shuffle_data_and_response(self, X, y):
        """Randomly permute the order of observations in the training data
        and response.
        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features)
            The training data for the optimization.
        y: ndarray, shape (n_samples, ).
            The training response for the optimization.
        Returns
        -------
        X: ndarray, shape (n_samples, n_features)
            The training data, with rows permuted in a random order.
        y: ndarray, shape (n_samples, ).
            The training response, premuted in a random order.  Note that the
            premutation used matches the permutation used to suffle the
            training data.
        """
        idxs = np.random.permutation(X.shape[0])
        X = X[idxs, :]
        y = y[idxs]
        return X, y

    def has_converged(self, cost, previous_cost):
        """Check if the gradient descent algorithm has converged.
        
        Parameters
        ----------
        cost: float
            The value of the cost function at this iteration of the gradient
            descent algorithm.
        previous_cost: float
            The value of the cost function at the previous iteration of the
            gradient descent algorithm.
        Returns
        -------
        has_converged: boolean
        """
        return self.step_size and np.abs(cost - previous_cost) <= self.step_size

    def has_converged_stochastic(self, window):
        """Check if the stochastic gradient descent algorithm has converged.
        The check here is a bit more delecate than in regular gradient descent.
        It does not make sense to check if the cost for a single data point to
        the next has stabilized, we need to average over a aggrigate of points.
        Parameters
        ----------
        window: positive integer
            The size of a window to check for the convergence criteria.
        Returns
        -------
        has_converged: boolean
        """
        if self.step_size:
            prior_sum = np.sum(self.cost_history[(-2 * window):(-window)])
            current_sum = np.sum(self.cost_history[(-window):])
            return np.abs(prior_sum - current_sum) <= self.step_size
        else:
            return False