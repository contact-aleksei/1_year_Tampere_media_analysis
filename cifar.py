import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.datasets import cifar10
import matplotlib.pyplot as plt

#import torchvision
#from torchvision import datasets
#from torch.utils.data import DataLoader
#import torch.nn as nn
# Declaration of Hidden Layers and Variables
# this is the size of our encoded representations
#encoding_dim = 128 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
                   # 64 floats -> compression of factor 12.25, assuming the input is 784 floats
                   # 128 floats -> compression of factor 6.125, assuming the input is 784 floats 
                   # 256 floats -> compression of factor 3.0625, assuming the input is 784 floats 
                   # 16 floats -> compression of factor  49.125, assuming the input is 784 floats
                   
                   #        uncompressed size      784 = 28*28
                   #ratio = ______ =   ______ = 6.125
                   #        compressed size          128
                   
# this is our input placeholder
input_img = Input(shape=(32, 32 , 3))

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation
#from keras.models import Model
#from keras import backend as K



(x_train, _), (x_test, _) = cifar10.load_data()

x = Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:])(input_img)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
x=UpSampling2D((4, 4))(x)
#x = Activation('relu')(x)
#x=UpSampling2D((3, 3))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = Activation('relu')(x)
#x = Conv2D(32, (3, 3), activation='relu')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation


# configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




import numpy as np

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format



from keras.callbacks import TensorBoard
autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
# Visualizing the reconstructed inputs and the encoded representations using Matplotlib
n = 4 # how many digits we will display

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    
    plt.imshow(x_test[i].reshape(32, 32, 3)) 
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(32, 32 , 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()