# -*- coding: utf-8 -*-
"""GANs(MNIST) keras.ipynb



## 1. Import Essential Library
"""

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

if not os.path.exists('images'):
    os.mkdir('images')

"""## 2. Define hyper-parameters"""

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
epochs = 20000
batch_size=128
sample_interval=50
colab = False

"""## 3. Load MNIST Dataset and Prepare the target annotations for GAN"""

# Load the dataset
(X_train, _), (_, _) = mnist.load_data()
# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)
# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

"""## 4. Lets define a function called build_generator() which defines our generator structure. Look at my slide.
- Dense: units=256, input_dim=latent_dim
- LeakyReLU: alpha=0.2
- BatchNormalization: momentum=0.8
- Dense: units=512
- LeakyReLU: alpha=0.2
- BatchNormalization: momentum=0.8
- Dense: units=1024
- LeakyReLU: alpha=0.2
- BatchNormalization: momentum=0.8
- Dense: units=number of pixel in each image in MNIST, actiovation = 'tanh'
- Reshape: shape = img_shape
"""

def build_generator():
    model = Sequential()

    model.summary()
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

"""## 5. Lets define a function called build_discriminator() which defines our  discriminator structure
- Flatten: input_shape=img_shape
- Dense: units=512
- LeakyReLU: alpha=0.2
- Dense: units=256
- LeakyReLU: alpha=0.2
- Dense: units=1, activation = 'sigmoid'
"""

def build_discriminator():

    model = Sequential()
    
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(inputs=img, outputs=validity)

"""## 6. Which optimizer do we use"""

optimizer = Adam(0.0002, 0.5)

"""## 7. In this section, we are trying to define a model and create it using compile."""

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

"""## 8. Generator definition not creation"""

# Build the generator
generator = build_generator()

"""## 9. Define the combined model. A combination of discriminator and generator. Fill where question marks are located."""

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))

# The discriminator takes generated images as input and determines validity
validity = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, validity)


"""## 10. This is a tool for visualizing the performance of generator"""

def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    if(colab == False):
        fig.savefig("images/%d.png" % epoch)
    else:
        plt.show()
    plt.close()

"""## 11. Training loop"""

for epoch in range(epochs):

  
    idx = np.random.randint(0, X_train.shape[0], batch_size)


    noise = np.random.normal(0, 1, (batch_size, latent_dim))

  
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

 

    noise = np.random.normal(0, 1, (batch_size, latent_dim))


    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        sample_images(epoch)

