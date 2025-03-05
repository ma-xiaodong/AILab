from keras.models import load_model
from keras.preprocessing import image
from keras import models
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

if __name__ == '__main__':
    model = load_model('./models/cats_and_dogs_4000.h5')
    base_dir = '/media/mxd/Document/data/kagglecatsanddogs/PetImages_small'
    image_path = os.path.join(base_dir, 'test/dogs/4532.jpg')

    img = image.load_img(image_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor /= 255.
    img_tensor = np.expand_dims(img_tensor, axis=0)

    plt.imshow(img_tensor[0])
    plt.show()

    model.summary()
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activate_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activate_model.predict(img_tensor)

    '''
    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    plt.show()
    '''
    layer_idx = 0
    for activation in activations:
        _, _, _, channels = activation.shape
        for channel in range(5):
            plt.matshow(activation[0, :, :, channel], cmap='viridis')
            plt.savefig('./fig' + str(layer_idx) + '_' + str(channel) + '.png')
        layer_idx += 1

