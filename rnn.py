import tensorflow as tf
import rnn_cell as rc

def LSTM(cell, X):
    
    out = []

    units = cell.get_units()

    shape = X[0].get_shape()
    if shape.ndims != 2:
        raise ValueError("dim not 2: %d", shape.ndims)

    right_size = shape[1].value

    h = tf.matmul(X[0], tf.zeros([right_size, units]))
    c = tf.matmul(X[0], tf.zeros([right_size, units]))

    for x in X:
        h,c = cell.call(x,(h,c))
        out.append((h,c))

    return out