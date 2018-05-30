
# coding: utf-8

# In[25]:


from keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image
import glob 
import numpy as np 
import cv2
import os
from skimage.transform import resize
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import scipy.misc as misc
import skimage as sk
from scipy.misc import imsave

IMG_ORI_WIDTH = 250 # original width
IMG_ORI_HEIGHT = 250 # original height
IMG_HEIGHT = 256 # resized height
IMG_WIDTH = 256 # resized width
IMG_CHANNELS = 3 # 

training_model_save = 'training_model.h5'
test_masks_save = 'maxkim_test_masks_nolambda_e30_b20.npy'
num_epoch=30
num_batch=20

# <h1>Show Image Sample</h1>

# In[26]:


train_sample = misc.imread('data/training_images/train_img_1.jpg')
print("type: ", type(train_sample))
print("shape: ", train_sample.shape)
#plt.imshow(train_sample)


# <h1>Show Mask Sample</h1>

# In[27]:


mask_sample = misc.imread('data/training_masks/train_mask_1.jpg')
print("type: ", type(mask_sample))
print("shape: ", mask_sample.shape)
#plt.imshow(mask_sample, cmap='gray')


# <h1>Get and Resize Train Image</h1>

# In[28]:


image_list = []
train_image_path = 'data/training_images/'
for i in range(1, 1501):
    im = misc.imread(train_image_path + 'train_img_' + str(i) + '.jpg') # read as numpy
    # resize the image to 256 X 256
    #plt.imshow(im)
    im = resize(im, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant')
    image_list.append(np.array(im)) # append it to the list 


# In[29]:


print("type: ", len(image_list))
print("type: ", type(image_list))
print("list length: ", len(image_list))
print("element shape: ", image_list[0].shape)
#plt.imshow(image_list[0]) # display the first image


# <h1>Get and Resize Train Masks</h1>

# In[30]:


mask_list = []
train_mask_path = 'data/training_masks/'
for i in range(1, 1501):
    # opencv reads an image as 3 Channles (BGR) although mask is a grayscale image
    im = misc.imread(train_mask_path + 'train_mask_' + str(i) + '.jpg') # read as numpy
    # resize the image to 256 X 256
    im = resize(im, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant')
    # expand dimension (256, 256) to (256, 256, 1) to indicate that it has 1 channel for model.fit
    #im = np.expand_dims(im, 2) # 2 indicates 3rd dimension
    mask_list.append(np.array(im)) # append it to the list 


# In[31]:


print("type: ", type(mask_list))
print("list length: ", len(mask_list))
print("element shape: ", mask_list[0].shape)
#plt.imshow(np.squeeze(mask_list[0]), cmap='gray') # display the first image
mask_list[0]


# <h1>Add Validaion Data to Train Data</h1>
# <p>It is easier to just slipt train data to validation data than manage train set and validation set. If you want to use validation set separately for model.fit, you need to do do model.fit(validation_data = (x_val , y_val) ) # tuple of x val and y val</p>

# In[32]:


# adding images
val_img_path = 'data/validation_images/'
for i in range(1, 501):
    im = misc.imread(val_img_path + 'validation_img_' + str(i) + '.jpg') # read as numpy
    # resize the image to 256 X 256
    im = resize(im, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant')
    image_list.append(np.array(im)) # append it to the list 

# adding masks
val_mask_path = 'data/validation_masks/'
for i in range(1, 501):
    im = misc.imread(val_mask_path + 'validation_mask_' + str(i) + '.jpg') # read as numpy
    # resize the image to 256 X 256
    im = resize(im, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant')
    # expand dimension (256, 256) to (256, 256, 1) to indicate that it has 1 channel for model.fit
    #im = np.expand_dims(im, 2) # 2 indicates 3rd dimension
    mask_list.append(np.array(im)) # append it to the list 


# In[33]:


# Display the first validation image
#plt.imshow(image_list[1500]) # display the first image


# In[34]:


# Display the last validation image
#plt.imshow(image_list[1999]) # display the first image


# In[35]:


# Display the first validation mask
#plt.imshow(np.squeeze(mask_list[1500]), cmap='gray') # display the last image


# In[36]:


# Display the last validation mask
#plt.imshow(np.squeeze(mask_list[1999]), cmap='gray') # display the last image


# In[40]:


print(type(image_list))
print(type(mask_list[0]))
print(type(image_list))
print(type(mask_list[0]))
print(np.array(image_list).shape, np.array(mask_list).shape)
print(np.array(image_list[0]).shape)
print(np.array(mask_list[0]).shape)


# <h1>Get And Resize test images</h1>

# In[41]:


test_list = []
sizes_test = []
test_image_path = 'data/testing_images/'
for i in range(1, 928):
    im = misc.imread(test_image_path + 'test_img_' + str(i) + '.jpg') # read as numpy
    # resize the image to 256 X 256
    sizes_test.append([im.shape[0], im.shape[1]])
    im = resize(im, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant')
    test_list.append(np.array(im)) # append it to the list 


# In[45]:


# Display the first test image
#plt.imshow(test_list[49]) # display the first image
print(type(test_list))
print("train_masks type: ", type(test_list))


# In[46]:


# Display the last teset image
#plt.imshow(test_list[926]) # display the first image


# In[47]:


print(type(test_list))
print(type(test_list[0]))
print(np.array(test_list).shape)


# <h1>Create Keras Metric</h1>
# <p>It's used for the loss function. It's a customized loss function kind of</p>

# In[48]:


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# <h1>Build U-Net Model</h1>

# In[ ]:


# Build U-Net Model

# Build U-Net model
#inputs = Input((3, IMG_HEIGHT, IMG_WIDTH))
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x) (inputs)  # (lambda x: x/255) is for inputs that are not normalized

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()
print('Model Compiled')


# In[ ]:


model.fit(np.array(image_list),np.array(mask_list),validation_split=0.2,epochs= num_epoch,batch_size=num_batch)


# In[ ]:


# Predict on test
preds_test = model.predict(np.array(test_list), verbose=1)

# Threshold predictions
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# In[ ]:


# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    # resize the images (256, 256) back to (250, 250) which is the original size of the training masks
    # change dimension (n, m, 1) to (n, m) which is the original dimension, just simple gray scale
    # before fitting (n,m) was transformed to (n,m,1) for model.fit function to indicate it's a grayscale image
    # append it to the list
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),(sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))


# In[ ]:


np.save((test_masks_save) ,preds_test_upsampled)


