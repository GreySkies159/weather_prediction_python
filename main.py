import tensorflow._api.v2.compat.v1 as tf
from data_retriever import build_data_subset

tf.disable_v2_behavior()


# function to measure accuracy by comparing actual (model output) to expected (correct answer) labels
def measure_accuracy(actual, expected):
    num_correct = 0
    for i in range(len(actual)):
        # actual - model predicted
        actual_value = actual[i]
        expected_value = expected[i]
        # checking if highest value is in same position in predicted and actual value
        # if in same position- predicted correctly
        if actual_value[0] >= actual_value[1] and expected_value[0] >= expected_value[1]:
            num_correct += 1
        elif actual_value[0] <= actual_value[1] and expected_value[0] <= expected_value[1]:
            num_correct += 1
    return (num_correct / len(actual)) * 100


# input shape - model factor number
input_shape = 4
# epochs - number of training cycles
epochs = 20000

# splitting data for training and testing (roughly 70% for training)
x_train, y_train = build_data_subset('2020_weather_data_Vancouver.csv', 1, 193)
x_test, y_test = build_data_subset('2020_weather_data_Vancouver.csv', 194, 83)
print(len(y_train))
print(len(y_test))

# y = Wx+b

# Input node to feed in any number of data points for training/testing
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_shape], name='x_input')
# Input node to feed in any number of correct labels for training purposes (either [1 0]  or [0 1])
y_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y_input')

# Variable node to represent weights, initial value of a bunch of ones
W = tf.Variable(initial_value=tf.ones(shape=[input_shape, 2]), name='W')
# Variable node to represent biases, initial value of a bunch of ones
b = tf.Variable(initial_value=tf.ones(shape=[2]), name='b')

# Output node to perform the calculation and fit a line through input factors and output labels
y_output = tf.add(tf.matmul(x_input, W), b, name='y_output')

# Loss function to measure difference between expected (correct) and actual (model output) answers
loss = tf.reduce_sum(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_output)))
# Adam optimizer will attempt to minimize loss by adjusting variable values at learning rate of 0.001
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()

# Create the tensorflow session
session = tf.Session()
session.run(tf.global_variables_initializer())

tf.train.write_graph(session.graph_def, '.', 'weather_prediction.pbtxt', False)

# perform training
for i in range(epochs):
    session.run(optimizer, feed_dict={x_input: x_train, y_input: y_train})
    print(i)

saver.save(session, 'weather_prediction.ckpt')

# printing out accuracy of test and train data
print(measure_accuracy(session.run(y_output, feed_dict={x_input: x_train}), y_train))
print(measure_accuracy(session.run(y_output, feed_dict={x_input: x_test}), y_test))
