
from sklearn import datasets
import numpy as np
def loaddata():
    digits = datasets.load_digits()

    X = digits.data.T
    target = digits.target
    Y = np.zeros((10,len(target)))
    
    for i in range(len(target)):
        Y[target[i],i] = 1

    ntrain = 1500

    train_x = X[:,:ntrain]
    train_y = Y[:,:ntrain]

    test_x = X[:,ntrain:]
    test_y = Y[:,ntrain:]

    return train_x, train_y, test_x, test_y


# train_x, train_y, test_x, test_y = loaddata()

# print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
# print(train_y[:,1000:1008])