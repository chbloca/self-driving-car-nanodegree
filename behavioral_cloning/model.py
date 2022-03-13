import csv

# Extract data from the csv file
samples = []
with open('/home/workspace/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data into training set (80%) and validation set (20%) 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

# Python generator --> It prevents memory saturation by feeding Keras with fit_generator due to the high amount of input data
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            images = []
            steerings = []
            
            for batch_sample in batch_samples:
                name_center = '/home/workspace/data/IMG/' + batch_sample[0].split('/')[-1]
                img_center = plt.imread(name_center)
                steering_center = float(batch_sample[3])
                images.append(img_center)
                steerings.append(steering_center)
                
                correction = 0.2
                name_left = '/home/workspace/data/IMG/' + batch_sample[1].split('/')[-1]
                img_left = plt.imread(name_left)
                steering_left = steering_center + correction
                images.append(img_left)
                steerings.append(steering_left)
                
                name_right = '/home/workspace/data/IMG/' + batch_sample[2].split('/')[-1]
                img_right = plt.imread(name_right)
                steering_right = steering_center - correction
                images.append(img_right)
                steerings.append(steering_right)
                
                # Data augmentation
                images.append(np.fliplr(img_center))
                steerings.append(- steering_center)
                images.append(np.fliplr(img_left))
                steerings.append(- steering_left)
                images.append(np.fliplr(img_right))
                steerings.append(- steering_right)
          
            X_train = np.array(images)
            y_train = np.array(steerings)
            
            yield shuffle(X_train, y_train) # Hold the return of these values
                   
'''
measurements = []
images = []
with open('/home/workspace/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        steering_center = float(line[3])
                
        # create adjusted steering measurements for the side camera images
        correction = 0.2 # to be tuned
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        img_center = plt.imread(line[0])
        img_left = plt.imread(line[1])
        img_right = plt.imread(line[2])
        
        images.append(img_center)
        images.append(img_left)
        images.append(img_right)
        
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)
        
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(- measurement)

print(X_train.shape, y_train.shape)
'''

# Calling the generators
batch_size = 32
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Model definition
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((60, 25), (0, 0))))
model.add(Convolution2D(24, kernel_size = (5,5), strides = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, kernel_size = (5,5), strides = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, kernel_size = (5,5), strides = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Convolution2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch = np.ceil(len(train_samples) / batch_size),
                                     validation_data = validation_generator,
                                     validation_steps = np.ceil(len(validation_samples) / batch_size),
                                     epochs = 2)
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
# Display model configuration configuration
model.summary()
# Save the model
model.save('model.h5')

'''
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.show()
'''


