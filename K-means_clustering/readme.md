## K-means Clustering

### Pseudocode:
```
Choose k random points to be the centroid of the k clusters.
Repeat until clusters stop changing:
    For every data point in the dataset, determine which centroid it is closest
        to and assign it to that cluster.
    Update the centroids to be the new center of all the points in that cluster
        (just take the arithmetic mean)
```

### Distance matric:
Euclidean Distance
![enclidean](https://latex.codecogs.com/gif.latex?d(\mathbf{a},&space;\mathbf{b})&space;=&space;||\mathbf{a}&space;-&space;\mathbf{b}||&space;\&space;=&space;\sqrt{\sum&space;(a_i&space;-&space;b_i)^2})


### How to run:
```
from kmean import Kmeans
model = Kmeans()
model.fit(X)
clusters = model.clusters
```

