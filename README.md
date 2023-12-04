# GRU_Cov_est

This project aims to developer a filter-like approach to compute a accurate state and covariance estimation when measurement covariance is unknown. The loss function used is exactly the same as the one shown in the paper (equ (9)), except the output is changed to error instead of the state to reduce the effect of different scales of data.

A few issues are observed in this model:

1. The convergence is not going well, oscillating is very common
2. Output format. Here i tried to use the single scan option of GRU, however, the end result, especially in the reuse part doesnt seem to match the loss.
3. 
