''' Reference: https://keras.io/ '''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import cv2
import os
from photutils.datasets import make_noise_image
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from statistics import mean
from skimage.restoration import denoise_nl_means, estimate_sigma
#%%
def preprocess(array):
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 64, 64))
    return array

def addNoise(array,noiseModel):
    if(noiseModel == 'gaussian'):
        intensity = 0.5
        gaussian = intensity*make_noise_image((64,64), distribution='gaussian', mean=0.,stddev=1.)
        noisy_array = gaussian + array
        return np.clip(noisy_array, 0.0, 1.0)
    
    elif(noiseModel == 'poisson'):
        intensity = 0.2
        lam = 1
        poissonNoise = intensity * np.random.poisson(lam, array.shape).astype(float)
        noisy_array = array + poissonNoise
        return np.clip(noisy_array, 0.0, 1.0)


def show(array1, array2, title):
    n = 10
    indices = np.array([i for i in range(n)])
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title)
    plt.show()

#%% Load Train data
SIZE = 64
train_data = []
path1 = r'C:\Users\keven\OneDrive - South Dakota State University - SDSU\CSCI 8920\Homework\Final Project 2.0\DataSets\Set 1\Train'
files=os.listdir(path1)
for i in files:
    img=cv2.imread(path1 + '/' + str(i), 0)   
    img=cv2.resize(img,(SIZE, SIZE))
    train_data.append(img)
train_data = np.array(train_data)
#%% Load Test data
test_data = []
path3 = r'C:\Users\keven\OneDrive - South Dakota State University - SDSU\CSCI 8920\Homework\Final Project 2.0\DataSets\Set 1\Test'
files=os.listdir(path3)
for i in files:
    img=cv2.imread(path3 + '/' + str(i), 0)   
    img=cv2.resize(img,(SIZE, SIZE))
    test_data.append(img)
test_data = np.array(test_data)
#%%
train_data = preprocess(train_data)
test_data = preprocess(test_data)

noisy_train_data = addNoise(train_data,'gaussian')
noisy_test_data = addNoise(test_data,'gaussian')
show(test_data, noisy_test_data, 'Row 1: Clean Test Images, Row 2: Noisy Test Images')
#%%
SIZE = 64
encoderInput = layers.Input(shape=(SIZE, SIZE, 1))

# Encoder
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoderInput)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# model
model = Model(encoderInput, x)
model.compile(optimizer="adam", loss="mse")
model.summary()

hist = model.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=100,
    batch_size=10,
    verbose = 1,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)

#%%
predictions = model.predict(noisy_test_data).reshape(test_data.shape)
show(noisy_test_data, predictions, 'Row 1: Noisy Test Images, Row 2: DAE Filtered Test Images')

#%%
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
#%%
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs=range(len(loss)) # Get number of epochs
plt.figure()
plt.plot(epochs, loss, label='Training Loss')
plt.title('Training & Validation loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
#%% Apply Median Filter
medianFiltered = []
for i in noisy_test_data:
    medianFiltered.append(ndimage.median_filter(i, size=(3,3)))
medianFiltered = np.array(medianFiltered)
show(noisy_test_data, medianFiltered, 'Row 1: Noisy Test Images, Row 2: Median Filtered Test Images')
#%% Apply NL means Filter
nlmFiltered = []
for i in noisy_test_data:
    nlmFiltered.append(denoise_nl_means(i, h=1.15 * estimate_sigma(i), fast_mode=False))
nlmFiltered = np.array(nlmFiltered)
show(noisy_test_data, nlmFiltered, 'Row 1: Noisy Test Images, Row 2: NL Means Filtered Test Images')

#%% Find SSIM
def SSIM(clean_array, noisy_array):
    indices = []
    for j in range(len(clean_array)):
        indices.append(ssim(clean_array[j], noisy_array[j],data_range=noisy_array[j].max() - noisy_array[j].min()))
    return indices
#%%
cleanSSIM = mean(SSIM(test_data,noisy_test_data))
daeSSIM = mean(SSIM(predictions,test_data))
medianFilteredSSIM = mean(SSIM(medianFiltered,test_data))
nlmFilteredSSIM = mean(SSIM(nlmFiltered,test_data))


    
