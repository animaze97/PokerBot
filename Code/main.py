import dataset_loader
import network

training_data = dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true.csv')
test_data = dataset_loader.loadDataTest('../Dataset/poker-hand-testing.csv')
# print training_data
# print "our: ", training_data
net = network.Network([85, 20, 10])

net.SGD(training_data=training_data, epochs=10, mini_batch_size=100, eta=0.25, test_data=test_data)