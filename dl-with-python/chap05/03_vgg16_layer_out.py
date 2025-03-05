from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pdb

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

    step = 1
    for i in range(50):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

if __name__ == '__main__':
    model = VGG16(weights='imagenet', input_shape = (150, 150, 3), 
                  include_top=False)
    model.summary()
    for i in range(10, 20):
        print('================ fig %d ================' %(i))
        plt.imshow(generate_pattern('block5_conv1', i))
        plt.show()

