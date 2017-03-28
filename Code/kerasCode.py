import keras_dataset_loader
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD
from keras import regularizers
import matplotlib.pyplot as plt
from keras.callbacks import Callback

training_input, training_output = keras_dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true copy.csv')
test_input, test_output = keras_dataset_loader.loadDataTest('../Dataset/poker-hand-testing copy.csv')


model = Sequential()
model.add(Dense(20, input_dim=85, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))


sgd = SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

test_cost = []
test_acc = []
train_cost = []
train_acc = []

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        # print model.get_weights()
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=1, batch_size = 10)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        print "LOGS ACC: ", str(logs['acc'])
        test_acc.append(logs['acc'])
        test_cost.append(loss)

# print test_output


history = model.fit(training_input, training_output, epochs=100, batch_size=10, callbacks=[TestCallback((test_input, test_output))])

# print model.summary()

plt.plot(history.history['loss'])
plt.title("Training Cost")
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.show()

plt.plot(history.history['acc'])
plt.title("Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()

plt.plot(test_cost)
plt.title("Testing Cost")
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.show()

plt.plot(test_acc)
plt.title("Testing Accuracy")
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.show()




# test_cost = []
# test_acc = []
# train_cost = []
# train_acc = []
# for i in range(0, 4):
#     print "EPOCH ", i
#     history = model.fit(training_input, training_output,epochs=1, batch_size=10)
#     scores = model.evaluate(test_input, test_output, batch_size=10, verbose=1)
#     train_cost.append(history.history['loss'][0])
#     train_acc.append(history.history['acc'][0])
#     test_cost.append(scores[0])
#     test_acc.append(scores[1])
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# plt.plot(train_cost)
# plt.title("Training Cost")
# plt.xlabel("Cost")
# plt.ylabel("Epochs")
# plt.show()
#
# plt.plot(train_acc)
# plt.title("Training Accuracy")
# plt.xlabel("Accuracy")
# plt.ylabel("Epochs")
# plt.show()
#
# plt.plot(test_cost)
# plt.title("Testing Cost")
# plt.xlabel("Cost")
# plt.ylabel("Epochs")
# plt.show()
#
# plt.plot(test_acc)
# plt.title("Testing Accuracy")
# plt.xlabel("Cost")
# plt.ylabel("Epochs")
# plt.show()



#
# print(history.history.keys())
# #  "Accuracy"
# plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# print scores
#
