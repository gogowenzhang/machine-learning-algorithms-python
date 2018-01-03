## KNN: non-parametric-classifier

### Pseudocode: 
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

![enclidean](https://latex.codecogs.com/gif.latex?d(\mathbf{a},&space;\mathbf{b})&space;=&space;||\mathbf{a}&space;-&space;\mathbf{b}||&space;\&space;=&space;\sqrt{\sum&space;(a_i&space;-&space;b_i)^2})

### Cosine Similarity

Angle

*Cosine* similarity is another commonly used distance metric. It's measuring the angle between the two vectors:

![cosine](https://latex.codecogs.com/gif.latex?d(\mathbf{a},&space;\mathbf{b})&space;=&space;\frac{\mathbf{a}&space;\cdot&space;\mathbf{b}}{||\mathbf{a}||&space;||\mathbf{b}||})

## To run this code
```
from KNearestNeighbors import KNearestNeighbors
knn = KNearestNeighbors(k=3, distance=euclidean_distance)
knn.fit(X, y)
y_predict = knn.predict(X_new)
```





