## Document classification with Naive Bayes

Likelihood function with Laplace smoothing: 

<img src="https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/img/likelihood.png" width='300' height='100'>

Posteriors function: 
<img src="https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/img/posterior.png" width='300' height='100'>



### How to run:
In terminal, you should be able to run:
```
from src.naive_bayes import NaiveBayes
my_nb = NaiveBayes()
my_nb.fit(X_train, y_train)
print 'Accuracy:', my_nb.score(X_test, y_test)
my_predictions =  my_nb.predict(X_test)
```


To run compare the result of my implementation and sklearn's implementation, in terminal: 
```
python src/run_naive_bayes.py
```

The dataset used in the comparison can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names)


Package Dependencies:
- numpy==1.13.1
- sklearn==0.18.1
