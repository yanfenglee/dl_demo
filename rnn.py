import tensorflow as tf
import rnn_cell as rc

def static_rnn(cell, X):
    
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

def static_bidirectional_rnn(fw_cell,bw_cell, X):
    state1 = static_rnn(fw_cell,X)

    X1 = list(reversed(X))
    state2 = static_rnn(bw_cell,X1)

    outputs = []
    for fw, bw in zip(state1, state2):
        h1,c1 = fw
        h2,c2 = bw
        outputs.append((h1+h2,c1+c2))

    return outputs