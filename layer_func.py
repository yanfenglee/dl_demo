import numpy as np

def softmax(Z):
    #ez = np.exp(Z - np.max(Z, axis = 0, keepdims=True))
    ez = np.exp(Z)
    A = ez/np.sum(ez, axis=0,keepdims=True)
    return A

def softmax_loss(AL, Y):
    aa = np.sum(Y * np.log(AL), axis=0, keepdims=True)
    loss = -np.mean(aa)
    return loss

def softmax_loss_backward(AL, Y):
    return AL - Y
