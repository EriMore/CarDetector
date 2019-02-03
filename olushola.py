# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:51:57 2018

@author: Engnr. Erioluwa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:08:17 2018

@author: Engnr. Erioluwa
"""

#import libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Convolution2D(filters = 64, kernel_size = (3 , 3), padding="same", data_format = 'channels_last', activation = 'relu', input_shape = (64, 64, 1)))
classifier.add(MaxPooling2D(pool_size = (2, 2), strides=2))

classifier.add(Convolution2D(filters = 64, kernel_size = (5 , 5), padding="same", activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3, 3), strides=2))

#classifier.add(Convolution2D(filters = 32, kernel_size = (5 , 5), padding="same", activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (3, 3), strides=2))
#
#classifier.add(Convolution2D(filters = 64, kernel_size = (5 , 5), padding="same", activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (3, 3), strides=2))


classifier.add(Flatten())

classifier.add(Dense(output_dim = 256, activation = 'relu', ))
#classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(output_dim = 16, activation = 'relu', ))
#classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(output_dim = 9, activation = 'softmax'))


classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])


from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical')


classifier.fit_generator(
        training_set,
        steps_per_epoch=3295,
        epochs=5,
        validation_data=test_set,
        validation_steps=1469)

  