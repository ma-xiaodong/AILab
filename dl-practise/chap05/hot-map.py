from keras import backend as K

layer_name = 'block5_conv1'
filter_index = 0
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, filter_index, :, :])
