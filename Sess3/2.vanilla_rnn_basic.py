from __future__ import print_function, division
import numpy as np
import tensorflow as tf

num_epochs = 6
total_series_length_train = 50000
total_series_length_test = 500
truncated_backprop_length = 16
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length_train//batch_size//truncated_backprop_length

def generateData():
    x_train = np.array(np.random.choice(2, total_series_length_train, p=[0.5, 0.5]))
    y_train = np.roll(x_train, echo_step)
    y_train[0:echo_step] = 0

    x_train = x_train.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y_train = y_train.reshape((batch_size, -1))

    return x_train, y_train

batchX_placeholder = tf.placeholder(tf.float32, [None, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [None, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [None, state_size])

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [-1, 1])
    # current_input = tf.transpose(current_input)
    input_and_state_concatenated = tf.concat([current_input, current_state], axis=1)  # Increasing number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

prediction_numbers = [tf.argmax(predictions_series_item, axis=1)[0] for predictions_series_item in predictions_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

x_train, y_train = generateData()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch_idx in range(num_epochs):
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x_train[:,start_idx:end_idx]
            batchY = y_train[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)

    seq_str = input("give me a sequence: ")
    seq = [int(item) for item in seq_str.split()]
    seq = np.lib.pad(seq, (0, truncated_backprop_length - np.shape(seq)[0] % truncated_backprop_length)
                    , 'constant', constant_values=(0))
    y = np.roll(seq, echo_step)
    _current_state = np.zeros((state_size))
    predictions_series, prediction_numbers = sess.run([predictions_series, prediction_numbers],
            feed_dict={
                batchX_placeholder: seq.reshape((-1, truncated_backprop_length)),
                batchY_placeholder: y.reshape((-1, truncated_backprop_length)),
                init_state: _current_state.reshape((-1, state_size))
            })


    print(" sequence: {}".format(seq))
    print("     gold: {}".format(y))
    print("predicted: {}".format(np.array(prediction_numbers)))

