## Adaboost 
### Key points
* Adaptive Boosting
* adds together many weak estimators
* at each step, each sample point is re-weighted based on whether it was correctly classified in the previous step
* each estimator also gets a weight depending on its misclassification rate


### Pseudocode:
![adaboost](https://github.com/gogowenzhang/machine-learning-algorithms-python/blob/master/img/adaboost.png)


### In terminal you should be able to run this code:
```
from boosting import AdaBoostBinaryClassifier
import numpy as np
from sklearn.model_selection import train_test_split

if __name__=='__main__':
   data = np.genfromtxt('data/spam.csv', delimiter=',')

   y = data[:, -1]
   X = data[:, 0:-1]

   X_train, X_test, y_train, y_test = train_test_split(X, y)

   my_ada = AdaBoostBinaryClassifier(n_estimators=50)
   my_ada.fit(X_train, y_train)
   print "Accuracy:", my_ada.score(X_test, y_test)
```
   
Package Dependencies:
- numpy==1.13.1
- sklearn==0.18.1
