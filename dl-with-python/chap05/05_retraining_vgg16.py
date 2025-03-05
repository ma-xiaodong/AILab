from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

def extract_features(conv_base, directory, batch_size, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))

    datagen = ImageDataGenerator(1./255)
    generator = datagen.flow_from_directory(
        directory, target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    print(directory, ': ', end='')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch

        i += 1
        if i * batch_size >= sample_count:
            break;

    print('\n')
    return features, labels

if __name__ == '__main__':
    conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(150, 150, 3))
    conv_base.summary()

    base_dir = '/media/mxd/Document/data/kagglecatsanddogs/PetImages_small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    print('======== extract featrues ========')
    train_features, train_labels = extract_features(
        conv_base, train_dir, 20, 2000)
    validation_features, validation_labels = extract_features(
        conv_base, validation_dir, 20, 1000)
    test_features, test_labels = extract_features(
        conv_base, test_dir, 20, 1000)

    print('======== retrain ========')
    assert len(train_features.shape)==4, 'Wrong shape of train_features'
    flatten_length = train_features.shape[1] * train_features.shape[2] * \
                     train_features.shape[3]
    train_features = np.reshape(
        train_features, (train_features.shape[0], flatten_length))
    validation_features = np.reshape(
        validation_features, (validation_features.shape[0], flatten_length))
    test_features = np.reshape(
        test_features, (test_features.shape[0], flatten_length))

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=flatten_length))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features, train_labels,
                        epochs=30, batch_size=20,
                        validation_data=(validation_features, 
                                         validation_labels))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

