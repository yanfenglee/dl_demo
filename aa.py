
from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()

X = digits.data.T
Y = digits.target.reshape(1,-1)

ntrain = 1500

train_x = X[:,:ntrain]
train_y = Y[:,:ntrain]

test_x = X[:,ntrain:]
test_y = (Y[:,ntrain:] == 4)

print(X.shape, Y.shape, train_x.shape,train_y.shape,test_x.shape,test_y.shape)

print(test_y)

def relu1(z):
    return np.maximum(z, 0, z)

def relu2(z):
    return z * (z > 0)

print(relu1(np.asarray([1,-1,2])))