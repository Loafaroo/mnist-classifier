import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import struct
import re
import os


def retrieve_data(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        size = struct.unpack(">I", f.read(4))  # size is number of elements
        size = size[0]                        # just get an integer instead of tuple
        
        if bool(re.search('idx1', filename)):
            nrows = 1
            ncols = 1
        elif bool(re.search('idx3', filename)):
            nrows, ncols = struct.unpack(">II", f.read(8))   # dimensions of data elements
        
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        ## get all the data as 8bit unsigned integers, big-endian byte order

    
    
    data = data.reshape(size, nrows, ncols)       
        
    return data

new_model = tf.keras.models.load_model('MNIST-model.h5')

file_images_test = 't10k-images.idx3-ubyte'
file_labels_test = 't10k-labels.idx1-ubyte'

images_test = retrieve_data(file_images_test)
labels_test = retrieve_data(file_labels_test)

images_test = images_test.reshape(*images_test.shape, 1)

images_test = images_test.astype('float32')

images_test /= 255

number_of_images = 10000

cwd = os.getcwd()

print(cwd)

for image_index in range(number_of_images):
    pred = new_model.predict(images_test[image_index].reshape(1, 28, 28, 1))
    predicted_label = pred.argmax()
    actual_label = labels_test[image_index][0][0]
    
    if not (predicted_label == actual_label ):
        
        fname = cwd + f"\\exceptions\\{image_index}-{predicted_label}-{actual_label}.png"
        print(f"predicted: {predicted_label} | actual: {actual_label} | index: {image_index}")
        plt.imsave(fname, images_test[image_index].reshape(28, 28), cmap="gray")

