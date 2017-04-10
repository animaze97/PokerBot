import tensorflow as tf
import keras_dataset_loader
import numpy as np

seed = 128
rng = np.random.RandomState(seed)


# training_input, training_output = \
#     datasetLoaderFor8Outputs.loadDataTrain('./POKER_data/original/poker-hand-training.csv')
# test_input, test_output = datasetLoaderFor8Outputs.loadDataTest('./POKER_data/original/poker-hand-testing.csv')

training_input, training_output = keras_dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true copy.csv')
test_input, test_output = keras_dataset_loader.loadDataTest('../Dataset/poker-hand-testing copy.csv')

def batch_creator(batch_size):
    """Create batch with random samples and return appropriate format"""
    # batch_mask = rng.choice( 1918, batch_size)

    # batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    # batch_x = preproc(batch_x)
    training_data = zip(training_input, training_output)
    np.random.shuffle(training_data)

    # training_input = unzipped[0]
    # training_output = unzipped[1]
    training_data = zip(*training_data)
    trainx = list(training_data[0])
    trainy = list(training_data[1])

    # mini_batches = [training_data[k:k + batch_size] for k in xrange(0,  1918, batch_size)]
    # mini_batches = zip(*mini_batches)
    batch_x = [trainx[k:k + batch_size] for k in xrange(0,  25010, batch_size)]
    batch_y = [trainy[k:k + batch_size] for k in xrange(0,  25010, batch_size)]

    # if dataset_name == 'train':
    #     batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
    #     batch_y = dense_to_one_hot(batch_y)

    return batch_x, batch_y

#set all variable
input_num_units = 85
hidden_num_units1 = 20
hidden_num_units2 = 20
#hidden_num_units3 = 20
#hidden_num_units4 = 20
output_num_units = 8

#define placeholders
x= tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

#set other variables

epochs = 3070
batch_size = 10
learning_rate = 0.02893

weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
    #'hidden3': tf.Variable(tf.random_normal([hidden_num_units2, hidden_num_units3], seed=seed)),
    #'hidden4': tf.Variable(tf.random_normal([hidden_num_units3, hidden_num_units4], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units2, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),
    #'hidden3': tf.Variable(tf.random_normal([hidden_num_units3], seed=seed)),
    #'hidden4': tf.Variable(tf.random_normal([hidden_num_units4], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_layer1 = tf.nn.relu(hidden_layer1)

hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
hidden_layer2 = tf.nn.relu(hidden_layer2)

#hidden_layer3 = tf.add(tf.matmul(hidden_layer2, weights['hidden3']), biases['hidden3'])
#hidden_layer3 = tf.nn.relu(hidden_layer3)

#hidden_layer4 = tf.add(tf.matmul(hidden_layer3, weights['hidden4']), biases['hidden3'])
#hidden_layer4 = tf.nn.relu(hidden_layer4)

output_layer = tf.matmul(hidden_layer2, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
# cost = tf.reduce_sum(tf.square(output_layer-y), name=None)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as sess:

    # create initialized variables
    sess.run(init)

    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize

    for epoch in range(epochs):
        avg_cost = 0.0
        total_batch = int( 25010 / batch_size)
        batch_x, batch_y = batch_creator(batch_size)
        for i in range(total_batch):
            # batch_x, batch_y = batch_creator(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x[i], y: batch_y[i]})

            avg_cost += c / total_batch

        print "Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost)
        # find predictions on val set
        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print "Validation Accuracy:", accuracy.eval({x: training_input, y: training_output})
    print "\nTraining complete!"

    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print "Validation Accuracy:", accuracy.eval({x: test_input, y: test_output})

    # predict = tf.argmax(output_layer, 1)
    # pred = predict.eval({x: test_x.reshape(-1, input_num_units)})