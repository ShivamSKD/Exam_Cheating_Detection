# ML Project to Detect Exam Cheating
 
## Introduction
 This project consists of a Machine Learning model which we will be using to detect whether or not students are cheating in exam or not.
 We take an image as an input and then do the prediction.
 The model is a _Convolutional Neural Network Model (CNN)_ having three sequential layers.
 The model is build using the python's **Keras Library with Tensorflow as a Backend** .
 
 ## Code
 ### Importing Libraries
 ```python
from keras.models import Sequential
from keras.layers import Convolution2D#since we are dealing with images we use convulational 2-d in case of videos we use 3-d
from keras.layers import MaxPooling2D
from keras.layers import Flatten#flattening: to convert large feature map into a vector of inputs for our cnn
from keras.layers import Dense#used to add fullly connected layers in an ann;
```
### Initialising the CNN
```python
classifier = Sequential()
```
### Convolution
```python
classifier.add(Convolution2D(32, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))
```
### Pooling
```python
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```
### Adding second and third Convolutional Layer
```python
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```
### Flattening
```python
classifier.add(Flatten())
```
### Full Connection
```python
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
```
### Compiling and Training our Model
```python
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/invigi_train',
                                                 target_size = (256, 256),
                                                 batch_size = 5,
                                                 class_mode = 'binary')
#input_shape = (256, 256, 3) so target image size is also 256x256
test_set = test_datagen.flow_from_directory('dataset/invi_test',
                                            target_size = (256, 256),
                                            batch_size = 1,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 290,
                         nb_epoch = 15,
                         validation_data = test_set,
                         nb_val_samples = 32)
```
### Training Output

- Epoch 1/15 
 58/58 [==============================] - 21s 355ms/step - loss: 0.7415 - accuracy: 0.4862 - val_loss: 0.6894 - val_accuracy: 0.5000 
- Epoch 2/15
58/58 [==============================] - 20s 338ms/step - loss: 0.6985 - accuracy: 0.5103 - val_loss: 0.6884 - val_accuracy: 0.5000
- Epoch 3/15
58/58 [==============================] - 20s 341ms/step - loss: 0.6933 - accuracy: 0.5034 - val_loss: 0.6917 - val_accuracy: 0.5312
- Epoch 4/15
58/58 [==============================] - 20s 345ms/step - loss: 0.6927 - accuracy: 0.5138 - val_loss: 0.6920 - val_accuracy: 0.5625
- Epoch 5/15
58/58 [==============================] - 21s 354ms/step - loss: 0.6967 - accuracy: 0.5241 - val_loss: 0.5226 - val_accuracy: 0.5000
- Epoch 6/15
58/58 [==============================] - 20s 342ms/step - loss: 0.6942 - accuracy: 0.5310 - val_loss: 0.6960 - val_accuracy: 0.5312
- Epoch 7/15
58/58 [==============================] - 20s 337ms/step - loss: 0.6871 - accuracy: 0.5103 - val_loss: 0.7229 - val_accuracy: 0.4688
- Epoch 8/15
58/58 [==============================] - 20s 339ms/step - loss: 0.6625 - accuracy: 0.6207 - val_loss: 0.7488 - val_accuracy: 0.4062
- Epoch 9/15
58/58 [==============================] - 20s 344ms/step - loss: 0.6623 - accuracy: 0.6103 - val_loss: 0.6132 - val_accuracy: 0.5625
- Epoch 10/15
58/58 [==============================] - 21s 359ms/step - loss: 0.6138 - accuracy: 0.6379 - val_loss: 2.1079 - val_accuracy: 0.5938
- Epoch 11/15
58/58 [==============================] - 21s 369ms/step - loss: 0.5954 - accuracy: 0.6655 - val_loss: 0.6922 - val_accuracy: 0.6562
- Epoch 12/15
58/58 [==============================] - 21s 360ms/step - loss: 0.5121 - accuracy: 0.7345 - val_loss: 1.7799 - val_accuracy: 0.5938
- Epoch 13/15
58/58 [==============================] - 21s 359ms/step - loss: 0.5584 - accuracy: 0.6897 - val_loss: 4.1728 - val_accuracy: 0.6250
- Epoch 14/15
58/58 [==============================] - 20s 341ms/step - loss: 0.5388 - accuracy: 0.7483 - val_loss: 0.0232 - val_accuracy: 0.6875
- Epoch 15/15
58/58 [==============================] - 20s 339ms/step - loss: 0.4725 - accuracy: 0.7897 - val_loss: 0.4520 - val_accuracy: 0.7188

### For Predicting validation Data
```python
import numpy as np
string1='validation/Validation_image ('
a=1
training_set.class_indices
for i in range(12):
    string2 = string1 + str(a) + ').jpg'
    img = image.load_img(string2, target_size = (256, 256))
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis = 0)
    result = np.round(np.clip(classifier.predict(test_image), 0, 1)).astype(bool)
    if result == False :
        print("Image"+str(a)+"is Cheating")
    else :
        print("Image"+str(a)+"is Not Cheating")
    a=a+1    
```
    
## Testing Our Model Using The above Code for predicting validation data

 ### Input Images
 ![alt text](https://i.postimg.cc/9FXWcj00/Validation-image-2.jpg)
- **Output:Not Cheating**






 ![alt text](https://i.postimg.cc/VkrPNs8f/Validation-image-10.jpg)
- **Output:Cheating**






 ![alt text](https://i.postimg.cc/Vkp19rR0/Validation-image-11.jpg)
- **Output:Cheating**






 ![alt text](https://i.postimg.cc/GpDRXL77/Validation-image-12.jpg)
- **Output: Cheating**
