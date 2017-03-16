# Accuracy Around 50%

# import dataset_loader
# import network
#
# training_data = dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true.csv')
# test_data = dataset_loader.loadDataTest('../Dataset/poker-hand-testing.csv')
# # print training_data
# # print "our: ", training_data
# net = network.Network([85, 20, 20, 20, 10])
#
# net.SGD(training_data=training_data, epochs=10, mini_batch_size=10, eta=3.0, test_data=test_data)
#

# Accuracy 95.1%
import dataset_loader
import network2

training_data = dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true.csv')
test_data = dataset_loader.loadDataTest('../Dataset/poker-hand-testing.csv')

net = network2.Network([85, 20, 20, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 3.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)