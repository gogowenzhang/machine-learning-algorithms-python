## KNN: non-parametric-classifier

Pseudocode: 
```
kNN:
    for every point in the dataset:
        calculate the distance between the point and x
    take the k points with the smallest distances to x 
    return the majority class among these items
```

## Distance Metrics
### Euclidean Distance

Straight line, L2

*Euclidean* distance is the distance metric you've probably heard of before:

$$ d(\mathbf{a}, \mathbf{b}) = ||\mathbf{a} - \mathbf{b}|| \ = \sqrt{\sum (a_i - b_i)^2} $$

### Cosine Similarity

Angle

*Cosine* similarity is another commonly used distance metric. It's measuring the angle between the two vectors:

$$ d(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}||  ||\mathbf{b}||} $$

## To run this code
```
from KNearestNeighbors import KNearestNeighbors
knn = KNearestNeighbors(k=3, distance=euclidean_distance)
knn.fit(X, y)
y_predict = knn.predict(X_new)
```





