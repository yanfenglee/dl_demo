import tensorflow as tf

class LSTMCell:
    def __init__(self, num_units):
        self.num_units = num_units
        self.igate = None
        self.fgate = None
        self.ggate = None
        self.ogate = None

    def get_units(self):
        return self.num_units

    def init_all_gates(self, input_size):
        self.igate = self.init_gate(input_size)
        self.fgate = self.init_gate(input_size)
        self.ggate = self.init_gate(input_size)
        self.ogate = self.init_gate(input_size)


    def init_gate(self, input_size):
        initializer = tf.orthogonal_initializer()
        Wx = tf.Variable(initializer([input_size,self.num_units]),trainable=True)
        Wh = tf.Variable(initializer([self.num_units,self.num_units]),trainable=True)
        b = tf.Variable(tf.zeros([self.num_units]),trainable=True)

        return Wx,Wh,b

    def linear(self,X,H,gate):
        wx,wh,b = gate
        return tf.matmul(X,wx) + tf.matmul(H,wh) + b

    def call(self, X, hc):
        h_prev,c_prev = hc

        if self.igate == None:
            self.init_all_gates(X.get_shape()[1].value)

        i = self.linear(X, h_prev, self.igate)
        f = self.linear(X, h_prev, self.fgate)
        o = self.linear(X, h_prev, self.ogate)
        g = self.linear(X, h_prev, self.ggate)

        c = c_prev * tf.sigmoid(f) + tf.tanh(g) * tf.sigmoid(i)
        h = tf.tanh(c) * tf.sigmoid(o)

        return h,c