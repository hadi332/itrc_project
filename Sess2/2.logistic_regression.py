import tensorflow as tf

# import mnist data

mnist_dir = "/home/hadi/Desktop/itrc workshop/day2/2. Mnist"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

lr = 0.01
num_epochs = 100
batch_size = 100

batch_number = mnist.train.num_examples//batch_size

# tf graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# variables: will be tuned by tensorflow
W = tf.Variable(tf.zeros(shape=[784, 10], dtype=tf.float32))
b = tf.Variable(tf.zeros(shape=[1, 10]))

# predicted distribution over classes
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# recognized digit in order to evaluate accuracy on test set
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cross entropy cost between predicted distribution and the gold one-hot vector
cost_tmp1 = y*tf.log(pred)
cost_tmp2 = -tf.reduce_sum(cost_tmp1, axis=1)
cost = tf.reduce_mean(cost_tmp2)

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

# dont forget to call that
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_cost = 0
        for b in range(batch_number):
            trainx, trainy = mnist.train.next_batch(batch_size)
            op, c = sess.run([optimizer, cost], feed_dict={x: trainx, y: trainy})
            total_cost+=c
        avg_cost = total_cost/batch_number
        print("epoch {}: {}".format(epoch, avg_cost))
    print("Optimization is finished...")

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("final accuracy on test set: {}".format(acc))

