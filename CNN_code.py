# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D#since we are dealing with images we use convulational 2-d in case of videos we use 3-d
from keras.layers import MaxPooling2D
from keras.layers import Flatten#flattening: to convert large feature map into a vector of inputs for our cnn
from keras.layers import Dense#used to add fullly connected layers in an ann

# Initialising the CNN
classifier = Sequential()

# 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))#32-no. of filters(feature maps we are using),3-rows feasture map,3-columns of feature map,256-size of input image 256X256 is the dimension of 2-d image in each channel 3-no. of channels for colored images we have 3 channels(RGB) and for B&W image we have only one channel since we are using tensorflow backend so our input sequence is (256,256,3) but for theano backend our input sequence will be (3,256,256)

#   2 - Pooling
#we apply max pooling   to reduce the size of our feature map and therefore to reduce the number of nodes in future fully connected layers and this will reduce complexity and time of execution but without loosing the performance.
#because we are keeping track of the parts of the image that
#contain the high numbers corresponding to where the feature detectors detected some specific features
#in the input image.
#So we don't lose the spatial structure information and therefore we don't lose the performance of the
#model.
#But at the same time we manage to reduce the time complexity and we make it less compute intensive.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
#this is done in the last   in order to improve the accuracy that we are getting from just one layer and also to decrease the overfitting problem.
#here we are not giving input_shape = (256, 256, 3) parameter because keras already knows that the input to this layer will be the output of previous two layers.
#input to this layer is the max_pool matrix that we obtained from previous layers
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#   3 - Flattening

classifier.add(Flatten())

#   4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))#128 is experimentation based no rules for that
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#rescale is for feature scaling(since pixel values range from 0-255 so we divide it by 255 to get the values in range of (0,1))
#above part is for image augmentation
#shear means that we are shifting each pixel of our image by some fixed proportion
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


#below is code for Validation
from keras.preprocessing import image
import numpy as np
string1='dataset/invi_validation/Validation_image ('
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
