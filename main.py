import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as dataset

mnist_data = dataset.read_data_sets("MNIST_Dataset/",one_hot=True)
input_neuron_layer_count = 784
hidden_neuron_count_1 = 1024
hidden_neuron_count_2 = 1024
hidden_neuron_count_3 = 1024
output_neuron_layer_count = 10
learning_rate = 1e-4
iteration_count = 1000
batch_size = 128
dropout = 0.5
X = tf.placeholder("float", [None, input_neuron_layer_count])
Y = tf.placeholder("float", [None, output_neuron_layer_count])
probability_dropout = tf.placeholder(tf.float32) 

w_tensor_1 = tf.Variable(tf.truncated_normal([input_neuron_layer_count, hidden_neuron_count_1], stddev=0.1))
w_tensor_2 = tf.Variable(tf.truncated_normal([hidden_neuron_count_1, hidden_neuron_count_2], stddev=0.1))
w_tensor_3 = tf.Variable(tf.truncated_normal([hidden_neuron_count_2, hidden_neuron_count_3], stddev=0.1))
w_tensor_out = tf.Variable(tf.truncated_normal([hidden_neuron_count_3, output_neuron_layer_count], stddev=0.1))
b_tensor_1 = tf.Variable(tf.constant(0.1, shape=[hidden_neuron_count_1]))
b_tensor_2 = tf.Variable(tf.constant(0.1, shape=[hidden_neuron_count_2]))
b_tensor_3 = tf.Variable(tf.constant(0.1, shape=[hidden_neuron_count_3]))
b_tensor_out = tf.Variable(tf.constant(0.1, shape=[output_neuron_layer_count]))
layer_1 = tf.add(tf.matmul(X,w_tensor_1), b_tensor_1)
layer_2 = tf.add(tf.matmul(layer_1, w_tensor_2), b_tensor_2)
layer_3 = tf.add(tf.matmul(layer_2, w_tensor_3), b_tensor_3)
layer_drop = tf.nn.dropout(layer_3, probability_dropout)
output_layer = tf.matmul(layer_3, w_tensor_out) + b_tensor_out
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(iteration_count):
    batch_x, batch_y = mnist_data.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, probability_dropout:dropout})

test_accuracy = sess.run(accuracy, feed_dict={X: mnist_data.test.images, Y: mnist_data.test.labels, probability_dropout:1.0})
print(test_accuracy)
