import keras_dataset_loader
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD
from keras import regularizers
import matplotlib.pyplot as plt

training_input, training_output = keras_dataset_loader.loadDataTrain('../Dataset/poker-hand-training-true copy.csv')
test_input, test_output = keras_dataset_loader.loadDataTest('../Dataset/poker-hand-testing copy.csv')


model = Sequential()
model.add(Dense(20, input_dim=85, activation='sigmoid', activity_regularizer=regularizers.l2(0.5)))
model.add(Dense(20, activation='sigmoid', activity_regularizer=regularizers.l2(0.5)))
model.add(Dense(20, activation='sigmoid', activity_regularizer=regularizers.l2(0.5)))
model.add(Dense(20, activation='sigmoid', activity_regularizer=regularizers.l2(0.5)))
model.add(Dense(10, activation='sigmoid', activity_regularizer=regularizers.l2(0.5)))

sgd = SGD(lr=3.0)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(training_input, training_output,epochs=300, batch_size=10)
scores = model.evaluate(test_input, test_output, batch_size=10, verbose=1)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

