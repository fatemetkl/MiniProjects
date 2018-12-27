# -*- coding: utf-8 -*-


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

"""### 2. Seting the hyper-parameters Value"""

batch_size = 128    # Here we define the batch size value
num_classes = 10    # Assign the number of class exists in MNIST dataset
epochs = 12       # Total Number of iteratin on mnist dataset 
img_rows, img_cols = 28, 28        # input image dimensions

"""### 3. Loading MNIST Dataset
Uncomment bellow line and use mnist.load_data() function to load mnist dataset
"""

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""### 4. This is an advance topic and not essential to be adept at this, but TAs know. Ask them!"""

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

"""### 5. Here we are going to understand:
1. How can we change variables type?
2. How can we normalized the numbers in a range between 0 and 1? This is a simple solution, But there are other ways. google it:)
3. How can we get the exact dimension of each variable
"""

x_train = x_train.astype('float32')      # This is the way we change the variable type
x_test = x_test.astype('float32')
x_train /= 255                           # Here we normalize the data between 0 and 1
x_test /= 255                            # Here we normalize the data between 0 and 1
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

"""### 6. Using this way, you can find which classes each data point belongs to!"""

print(y_train[0])  
print(y_train[10000])  
# This refers to that the sample #1 is related to Class 5

"""### 7.Convert target representation from a simple scalar to one-hot representation:
One-Hot encoding. A one hot encoding is a representation of categorical variables as binary vectors. Each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.
"""

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train[0])    # above command assign 1 to 5th element of a vector and others have value 0.



model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D( pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

"""### 9. Same as before, we need to compile our above model with an optimizer and a caregorical loss function. You already know it."""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

"""### 10. Here, we try to fit our model on MNIST Dataset"""

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""## Pre-trained models and Data Augmentation:
In this section, we are going to use a pre-trained model called MobileNet. Using this network and a new dataset, we train a newly defined Convolutinal Neural Network. A comprehensive list of goals are presented bellow (Be cautious of What I list. Think about them)

1. Finetuning A pre-trained deep neural network
2. Training on a new dataset gave you before handson.
3. Data Augmenting using KERAS Utilities.
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

"""### 0. Hyper-parameters definition"""

img_width, img_height = 128, 128    # input image has size (3, 128, 128)
train_data_dir = "data/train"    # Location of training data
validation_data_dir = "data/val"    # Location of validation data
nb_train_samples = 244       # Total Number of Training samples
nb_validation_samples = 153       # Total Number of Validations samples
batch_size = 16
epochs = 50

"""### 1.Using commands introduced in hands on CNN, try to load MobileNet instead of VGG19. Just change the name ;)"""

model.application

model.summary()

"""### 2. try to freeze just all of the layers in model included above. look at slides if you need"""

# Freeze the first five layers which you don't want to train. 
for layer in model.layers[5]:   ######## You shold change this line ########
    layer.trainable = False

"""### 3. Here we are going to attach the new classifier at the end of pretrained model. This is a new technique whcih we are going to explore more.
- Flatten Layer
- Desne Layer: units: 1024, activation = "relu"
- Dropout: rate = 0.5
- Desne Layer: units: 512, activation = "relu"
- Desne Layer: units: 2, activation = "softamax"
"""

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

model_final.summary()

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

"""### 4. Define a data augmentator as presented in CNN hands-on slides according to the following parameters:
- rescale = 1./255
- horizontal_flip = True
- fill_mode = "nearest"
- zoom_range = 0.6
- width_shift_range = 0.2
- height_shift_range=0.4
- rotation_range=25
"""

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.3,
                    width_shift_range = 0.3,
                    height_shift_range=0.3,
                    rotation_range=30)

# This an augmentator for test dataset
test_datagen = ImageDataGenerator(rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.3,
                    width_shift_range = 0.3,
                    height_shift_range=0.3,
                    rotation_range=30)



train_generator = train_datagen.flow_from_directory(train_data_dir,
                        target_size = (img_height, img_width),
                        batch_size = batch_size, 
                        class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                        target_size = (img_height, img_width),
                        class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
model_final.fit_generator(train_generator,
        samples_per_epoch = nb_train_samples,
        epochs = epochs,
        validation_data = validation_generator,
        nb_val_samples = nb_validation_samples,
        callbacks = [checkpoint, early])

