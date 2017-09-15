import tensorflow as tf
from tensorflow.python import debug as tf_debug

class LSTMCell:
    def __init__(self, input_size, num_units):
        self.input_size = input_size
        self.num_units = num_units

        self.igate = self.init_gate()
        self.fgate = self.init_gate()
        self.ggate = self.init_gate()
        self.ogate = self.init_gate()

    def get_state_size(self):
        return (self.input_size,self.num_units)

    def init_gate(self):
        initializer = tf.orthogonal_initializer()
        Wx = tf.Variable(initializer([self.input_size,self.num_units]),trainable=True)
        Wh = tf.Variable(initializer([self.num_units,self.num_units]),trainable=True)
        b = tf.Variable(tf.zeros([self.num_units]),trainable=True)

        return Wx,Wh,b

    def linear(self,X,H,gate):
        wx,wh,b = gate
        return tf.matmul(X,wx) + tf.matmul(H,wh) + b

    def call(self, X, hc):
        h_prev,c_prev = hc

        i = self.linear(X, h_prev, self.igate)
        f = self.linear(X, h_prev, self.fgate)
        o = self.linear(X, h_prev, self.ogate)
        g = self.linear(X, h_prev, self.ggate)

        c = c_prev * tf.sigmoid(f) + tf.tanh(g) * tf.sigmoid(i)
        h = tf.tanh(c) * tf.sigmoid(o)

        return h,c


def RNN(cell, X):
    
    out = []

    input_size,units = cell.get_state_size()

    h = tf.zeros([128, units])
    c = tf.zeros([128, units])

    for x in X:
        h,c = cell.call(x,(h,c))
        out.append((h,c))

    return out

def procdata(X, num_steps, input_size):
    X = tf.transpose(X,[1,0,2])
    X = tf.reshape(X,[-1, input_size])
    X = tf.split(X, num_steps)
    return X

from tensorflow.examples.tutorials.mnist import input_data

def main():

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    learning_rate = 0.001
    max_epoch = 10000
    batch_size = 128
    display_step = 100

    n_input = 28
    n_steps = 28
    n_hidden = 128
    n_classes = 10

    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    biases = tf.Variable(tf.random_normal([n_classes]))

    # begin lstm
    x1 = procdata(x,num_steps=n_steps,input_size=n_input)
    cell = LSTMCell(input_size=n_input, num_units=n_hidden)
    outputs = RNN(cell, x1)
    h,_ = outputs[-1]
    #outputs = tf.reshape(outputs, [-1,n_hidden])
    yhat = tf.matmul(h, weights) + biases

    predict = tf.nn.softmax(yhat)
    
    # cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat,labels=y))

    # optimize
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # compute accuracy
    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        epoch = 1
        while epoch <= max_epoch:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if epoch % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print ("epoch " + str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
            
            epoch += 1
        print ("Optimization Finishes!")

        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print ("Testing accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


main()