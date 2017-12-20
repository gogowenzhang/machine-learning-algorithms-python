### Decision Tree pseudocode
```
function BuildTree:
    If every item in the dataset is in the same class
    or there is no feature left to split the data:
        return a leaf node with the class label
    Else:
        randomly choose a subset of the features
        find the best feature and value to split the data
        split the dataset
        create a node
        for each split
            call BuildTree and add the result as a child of the node
        return node
```

### Random Forest pseudocode
```
Repeat num_trees times:
     Create a random sample of the data with replacement
     Build a decision tree with that sample
 Return the list of the decision trees created
 ```
 
 ### Run this code in terminal
 ```
from RandomForest import RandomForest
rf = RandomForest(num_trees=10, num_features=2)
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
print "score:", rf.score(X_test, y_test)
```
Dependencies:
numpy==1.13.1
