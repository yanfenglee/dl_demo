import numpy as np
import matplotlib.pyplot as plt
import h5py
from layer_func import *
from data_utils import *

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1])*3.5
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def relu_farward(z):
    out = np.maximum(0,z)
    cache = z
    return out, cache

def linear_forward(A_prev, W, b):
    Z = np.dot(W,A_prev) + b
    return Z,(A_prev, W, b)

def linear_activation_forward(A_prev, W, b):
    
    Z, linear_cache = linear_forward(A_prev, W, b)

    A, activation_cache = relu_farward(Z)

    return A, (linear_cache, activation_cache)

def L_model_forward(X, parameters):
    layer_caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        layer_caches.append(cache)

    ZL, linear_cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    AL = softmax(ZL)
    layer_caches.append(linear_cache)

    return AL, layer_caches

def compute_cost(AL, Y):
    cost = softmax_loss(AL, Y)
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    return dA * (Z>0)

def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache
    
    dZ = relu_backward(dA, activation_cache)
    
    return linear_backward(dZ, linear_cache)

def L_model_backward(AL, Y, layer_caches):
    grads = {}
    L = len(layer_caches) # the number of layers
    
    dZL = softmax_loss_backward(AL, Y)

    current_cache = layer_caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZL, current_cache)
    
    for l in reversed(range(L-1)):
        
        current_cache = layer_caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
        
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        AL, layer_caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL,Y,layer_caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            #if i in [0,1,10,100,1000,10000,30000]:
                #print("al: ", AL[:,:5])

        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def test(test_x, test_y, parameters):
    yhat, cache = L_model_forward(test_x, parameters)
    
    n = 15
    print('----target-----')
    print(test_y[:,:n])
    #print(yhat[:,:n])

    yhat = (yhat>0.5)-0
    print('----predict-----')
    print(yhat[:,:n])
    yhat[yhat==0] = -1

    result = (test_y == yhat)-0
    p = np.sum(result) / (np.sum((yhat==1))+1)

    print("precision: ", p,np.sum(result), np.sum((yhat==1)))

def main():
    train_x,train_y,test_x,test_y = loaddata()

    train_x = train_x/255.
    test_x = test_x/255.

    print(train_x.shape,test_x.shape)
    
    layers_dims = [64, 40, 60, 40, 10]
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075,num_iterations = 5000, print_cost = True)
    
    test(test_x, test_y, parameters)

main()