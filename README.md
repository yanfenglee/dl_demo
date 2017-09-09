# dl_demo
* asdf what is that of coz


$$ \frac{\partial L}{\partial o_i}=-\sum_ky_k\frac{\partial \log p_k}{\partial o_i}=-\sum_ky_k\frac{1}{p_k}\frac{\partial p_k}{\partial o_i}\\=-y_i(1-p_i)-\sum_{k\neq i}y_k\frac{1}{p_k}({\color{red}{-p_kp_i}})\\=-y_i(1-p_i)+\sum_{k\neq i}y_k({\color{red}{p_i}})\\=-y_i+\color{blue}{y_ip_i+\sum_{k\neq i}y_k({p_i})}\\=\color{blue}{p_i\left(\sum_ky_k\right)}-y_i=p_i-y_i $$