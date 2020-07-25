import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

file_images_train = 'train-images.idx3-ubyte'
file_labels_train = 'train-labels.idx1-ubyte'
file_images_test = 't10k-images.idx3-ubyte'
file_labels_test = 't10k-labels.idx1-ubyte'

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

        print (nrows, ncols) ## BUG: for filename = 'train-labels.idx1-ubyte'

        
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        ## get all the data as 8bit unsigned integers, big-endian byte order

    
    input()
    data = data.reshape(size, nrows, ncols)       
        
    return data

def main():
    images_train = retrieve_data(file_images_train)
    labels_train = retrieve_data(file_labels_train)
    images_test = retrieve_data(file_images_test)
    labels_test = retrieve_data(file_labels_test)

    images_train = images_train.reshape(*images_train.shape, 1)
    images_test = images_test.reshape(*images_test.shape, 1)

    input_shape = (28, 28, 1)
    
    # Making sure that the values are float so that we can get decimal points after division
    images_train = images_train.astype('float32')
    images_test = images_test.astype('float32')
    
    # Normalizing the RGB codes by dividing it to the max RGB value.
    images_train /= 255
    images_test /= 255

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=images_train,y=labels_train, epochs=10)

    model.evaluate(x_test, y_test)

if __name__ == '__main__':
    main()


