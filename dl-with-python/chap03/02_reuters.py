import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
np.set_printoptions(threshold = np.inf)
#print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
#print(type(train_data[0]))

def translate(sentense):
    word_index = reuters.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded = ' '.join([reverse_index.get(i - 3, '*') for i in sentense])
    return decoded

def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = vectorize_sequences(train_labels, 46)
one_hot_test_labels = vectorize_sequences(test_labels, 46)

print(one_hot_train_labels.shape, one_hot_test_labels.shape)

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train,
                    epochs = 20, batch_size = 512,
                    validation_data = (x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

result = model.evaluate(x_test, one_hot_test_labels)
print(result)

