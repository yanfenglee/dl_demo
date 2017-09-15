import tensorflow as tf
import rnn_cell as rc
import rnn

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
    cell = rc.LSTMCell(n_hidden)
    outputs = rnn.LSTM(cell, x1)
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

        test_len = 512
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print ("Testing accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


main()