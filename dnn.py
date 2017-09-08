import numpy as np
import matplotlib.pyplot as plt
import h5py

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) # * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def sigmoid_farward(z):
    out = 1/(1+np.exp(-z))
    cache = out
    return out, cache

def relu_farward(z):
    out = np.maximum(0,z)
    cache = z
    return out, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    Z = np.dot(W,A_prev) + b

    if activation == "sigmoid":
        A, activation_cache = sigmoid_farward(Z)
    elif activation == "relu":
        A, activation_cache = relu_farward(Z)

    linear_cache = (A_prev, W, b)

    return A, (linear_cache, activation_cache)

def L_model_forward(X, parameters):
    layer_caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        layer_caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    layer_caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, layer_caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))/m
    
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

def sigmoid_backward(dA, cache):
    A = cache
    return dA * A * (1-A)

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    return linear_backward(dZ, linear_cache)

def L_model_backward(AL, Y, layer_caches):
    grads = {}
    L = len(layer_caches) # the number of layers
    #m = AL.shape[1]
    #Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y,1-AL))

    current_cache = layer_caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):
        
        current_cache = layer_caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
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
            #if i in [1,10,100,1000,10000,30000]:
                #print(grads)

        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

from sklearn import datasets
def loaddata():
    digits = datasets.load_digits()

    X = digits.data.T
    Y = digits.target.reshape(1,-1)

    ntrain = 1500

    train_x = X[:,:ntrain]
    train_y = (Y[:,:ntrain] == 4)-0

    test_x = X[:,ntrain:]
    test_y = (Y[:,ntrain:] == 4)-0

    return train_x, train_y, test_x, test_y

def loadcat():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def test(test_x, test_y, parameters):
    yhat, cache = L_model_forward(test_x, parameters)
    
    print('----target-----')
    print(test_y)

    yhat = (yhat>0.5)-0
    print('----predict-----')
    print(yhat)
    yhat[yhat==0] = -1

    result = (test_y == yhat)-0
    p = np.sum(result) / (np.sum((yhat==1)+1))

    print("precision: ", p,np.sum(result), np.sum((yhat==1)))

def main():
    train_x,train_y,test_x,test_y = loaddata()

    train_x = train_x/255.
    test_x = test_x/255.

    print(train_x.shape,test_x.shape)
    
    layers_dims = [64, 10, 17, 15, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075,num_iterations = 2500, print_cost = True)
    
    test(test_x, test_y, parameters)


def main2():
    train_x_orig, train_y, test_x_orig, test_y, classes = loadcat()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    layers_dims = [12288, 20, 17, 15, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075,num_iterations = 2000, print_cost = True)
    
    test(test_x, test_y, parameters)

#main()
main2()