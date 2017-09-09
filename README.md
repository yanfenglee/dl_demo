# dl_demo

## softmax function

$$a_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

## cross entropy loss

$$L = -\sum_k y_k \log a_k$$

## cross entropy loss derivative

$$\frac{\partial L}{\partial z_i}=-\sum_k y_k \frac{1}{a_k} \frac{\partial a_k}{\partial z_i}$$
$$= -y_i(1-a_i)-\sum_{k\neq i} y_k \frac{1}{p_k}(-p_ka_i)$$
$$= -y_i(1-a_i)+\sum_{k\neq i} y_k a_i$$
$$= -y_i+y_ia_i+\sum_{k\neq i} y_k a_i$$
$$= a_i \sum_ky_k -y_i$$
$$= a_i-y_i$$



