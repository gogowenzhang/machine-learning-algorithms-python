## NMF: Non-Negative Matrix Factorization

There are a couple ways to sovling NMF. This implementaion adapted Alternating Least Squares (ALS) algorithm. 

We are trying to find matrics, W and H such that:
            V ~ W * H       w_i,j & h_i,j >= 0

### Pseudocode of ALS:
```
Initialize W to small, positive, random values.
For max number of iterations:
    find the least squres solution to V = W * H w.r.t H.
    clip negative values in H to 0: H < 0 =0
    find the least squeres solution to V = W * H w.r.t W.
    clip negative values in W to 0: W < 0 = 0.
```


### How to run:
```
from nmf import NMF
factorizer = NMF(k=7, max_iters=35, alpha=0.5)
W, H = factorizer.fit(X, verbose=True)
print 'reconstruction error:', factorizer.reconstruction_error
```
