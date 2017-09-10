# dl_demo

## softmax function

$$a_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

## derivative of softmax

$$ \frac{\partial a_j}{\partial z_i} = a_i(1 - a_i), \quad i = j $$
$$ \frac{\partial a_j}{\partial z_i} = -a_i a_j,\quad i \neq j. $$

## cross entropy loss

$$L = -\sum_k y_k \log a_k$$

## cross entropy loss derivative

$$ \frac{\partial L}{\partial z_i} = -\sum_k y_k \frac{1}{a_k} \frac{\partial a_k}{\partial z_i} $$
$$ = -y_i(1-a_i) - \sum_{k\neq i} y_k \frac{1}{a_k}(-a_ka_i) $$
$$ = -y_i(1-a_i) + \sum_{k\neq i} y_k a_i $$
$$ = -y_i+y_ia_i + \sum_{k\neq i} y_k a_i $$
$$ = a_i \sum_ky_k - y_i $$
$$ = a_i-y_i $$





