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
import overfitting

# training_data = dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true.csv')
# test_data = dataset_loader.loadDataTest('../Dataset/poker-hand-testing.csv')

# net = network2.Network([85, 20, 20, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 100, 10, 0.5,lmbda=0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True
#         , monitor_training_accuracy=True, monitor_training_cost=True)

overfitting.main(filename='results.txt', num_epochs=100,
         training_cost_xmin=200,
         test_accuracy_xmin=200,
         test_cost_xmin=0,
         training_accuracy_xmin=0,
         training_set_size=25000,
         lmbda=0.5)