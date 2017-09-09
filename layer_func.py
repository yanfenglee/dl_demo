import numpy as np

def softmax_loss(ZL, Y):
    ez = np.exp(ZL - np.max(ZL, axis = 1, keepdims=True))
    AL = ez/np.sum(ez, axis=1,keepdims=True)
    return -np.mean(np.sum(Y * np.log(AL), axis=1, keepdims=True))

def softmax_loss_backward(AL, Y):
    return AL - Y
