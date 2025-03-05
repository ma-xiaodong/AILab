import os
import cv2
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pdb

def decode_predicts(preds):
    assert preds.shape == (1, 1), 'Error predicts shape!'
    if preds[0][0] <= 0.5:
        return 'cat'
    else:
        return 'dog'

def write_heatmap(model, img, ori_cls, idx):
    # preprocess the image
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.

    # predict the classification
    preds = model.predict(img_tensor)
    pre_cls = decode_predicts(preds)

    # compute the gradients
    pred_output = model.output[:, 0]
    last_conv_layer = model.get_layer('conv2d_4')
    grads = K.gradients(pred_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img_tensor)

    # get the heat map and show it
    for i in range(last_conv_layer.output.shape[3]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # show the original image mixed with heatmap
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite(str(idx) + '_orig_' + ori_cls + '.jpg', img)
    cv2.imwrite(str(idx) + '_pred_' + pre_cls + '.jpg', superimposed_img)

if __name__ == '__main__':
    # load model
    model = load_model('./models/cats_and_dogs_4000.h5')

    # load the image
    base_dir = '/media/mxd/Document/data/kagglecatsanddogs/PetImages_small'

    for i in range(4520, 4530):
        file_name = 'test/dogs/' + str(i) + '.jpg'
        image_path = os.path.join(base_dir, file_name)
        img = image.load_img(image_path, target_size=(150, 150))
        write_heatmap(model, img, 'dog', i)

