import tensorflow as tf

mnist_dir = "/home/hadi/Desktop/itrc workshop/day2/2. Mnist"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

num_epochs = 10
batch_size = 50

batch_number = mnist.train.num_examples//batch_size

x = tf.placeholder(tf.float32, shape=[None, 784], name="moteghayyer1")
y_gold = tf.placeholder(tf.float32, shape=[None, 10])

x_img = tf.reshape(x, shape=[-1, 28, 28, 1])

with tf.name_scope('conv1'):
    # to be used in conv2d: [filter_height, filter_width, in_channels, out_channels]
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], mean=0, stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1_tmp = tf.nn.conv2d(x_img, filter=W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1 # SAME: outpus=ceil[input/stride] or VALID: output = floor[(input - filter)/Stride] + 1
    h_conv1 = tf.nn.relu(h_conv1_tmp)

with tf.name_scope('pool1'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('conv2'):
    # to be used in conv2d: [filter_height, filter_width, in_channels, out_channels]
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], mean=0, stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2_tmp = tf.nn.conv2d(h_pool1, filter=W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
    h_conv2 = tf.nn.relu(h_conv2_tmp)

with tf.name_scope('pool2'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

with tf.name_scope('fc1'):
    w_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024]))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

with tf.name_scope('fc2'):
    w_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, 10]))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

    y_conv = tf.matmul(h_fc1, w_fc2) + b_fc2

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_gold)
    cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_prediction = tf.cast(tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_gold, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        total_cost = 0
        for b in range(batch_number):
            trainx, trainy = mnist.train.next_batch(batch_size)
            op, c = sess.run([optimizer, cost], feed_dict={x: trainx, y_gold: trainy})
            total_cost+=c
        avg_cost = total_cost/batch_number
        print("epoch {}: {}".format(epoch, avg_cost))
    print("Optimization is finished...")

    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_gold: mnist.test.labels})
    print("final accuracy on test set: {}".format(acc))





input()
