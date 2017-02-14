import cv2
import csv
import numpy as np
import pandas
import sklearn

from random import shuffle

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Activation, Input, Embedding, Lambda, LSTM, Dense, merge, Convolution2D, MaxPooling2D, Reshape, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras import initializations
from sys import getsizeof
# beneficios novia paisa



# function to get list of images and list of steering angles
def get_data_list(list_log_file):
    list_path=[]
    list_steering = []
    #df=pandas.read_csv(list_log_file[0])
    #print(df[])
    #list_path = list(df['center'])
    #list_steering = list(df['steering'])
    df=pandas.read_csv(list_log_file[0],header=None) 
    list_path = list_path + list(df[0])
    list_steering = list_steering + list(df[3])
    return list_path, list_steering


# function to get a batch of images or the hall dataset
def get_batch_images(list_images, flag_normal = True, batch_size=-1):
    img_test = cv2.imread(list_images[0])
    hh, ww, ch = img_test.shape
    if batch_size == -1:
        #batch_data = np.zeros(shape=(len(list_images), hh, ww, ch), dtype=np.float32)
        batch_data = np.zeros(shape=(len(list_images), 66, 200, ch), dtype=np.float32)
        k = 0
        for im_path in list_images:
            im_path = im_path.replace("darial","autti")
            img_to = cv2.imread(im_path)
            img_to = cv2.cvtColor(img_to, cv2.COLOR_BGR2RGB)
            batch_data[k] = cv2.resize(img_to, (200,66))
            k += 1
        batch_data /= 255
        batch_data -= 0.5
        print("memory",getsizeof(batch_data))
    else:
        batch_data = np.zeros(shape=(batch_size, hh, ww, ch), dtype=np.float32)
        for k in range(batch_size):
            batch_data[k] = cv2.imread(list_images[k])
    return batch_data


# generator to create random batches
def generator_batch_images(list_images, batch_size=4):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        offset_angle = 0.23
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                lname = './IMG/'+batch_sample[1].split('/')[-1]
                rname = './IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name)
                left_image = cv2.imread(lname) 
                right_image = cv2.imread(rname)
                #asd = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                #print("max min",np.max(asd),np.min(asd))
                if batch_sample[3] != "steering":
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle 

                    # data augmentation
                    center_image = center_image[20:150]
                    center_image = cv2.resize(center_image, (200,66))
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                    center_image = center_image.astype(np.float32)

                    left_image = left_image[20:150]
                    left_image = cv2.resize(left_image, (200,66))
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                    left_image = left_image.astype(np.float32)

                    right_image = right_image[20:150]
                    right_image = cv2.resize(right_image, (200,66))
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                    right_image = right_image.astype(np.float32)


                    # normalization
                    center_image /= 255
                    center_image -= 0.5 
                    flipped_center_image = np.copy(center_image)
                    flipped_center_image = np.fliplr(flipped_center_image)

                    images.append(center_image)
                    angles.append(center_angle)

                    images.append(left_image)
                    angles.append(center_angle + offset_angle)
                    images.append(right_image)
                    angles.append(center_angle - offset_angle)
                    images.append(flipped_center_image)
                    angles.append(-1*center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print("shape",X_train.shape[0],y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)


# read csv file
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# split data train and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create generators
train_generator = generator_batch_images(train_samples, batch_size=4)
validation_generator = generator_batch_images(validation_samples, batch_size=4)


# --create architecture--
main_input = Input(shape=(66, 200, 3), name='main_input')

conv1 = Convolution2D(16, 5, 5, activation='relu', border_mode='valid')(main_input)
conv2 = Convolution2D(24, 5, 5, activation='relu', border_mode='valid')(conv1)

conv3 = Convolution2D(24, 5, 5, activation='relu', border_mode='valid')(conv2)
conv4 = Convolution2D(48, 3, 3, activation='relu', border_mode='valid')(conv3)
conv5 = Convolution2D(48, 3, 3, activation='relu', border_mode='valid')(conv4)

flat1 = Flatten()(conv5)

fc1 = Dense(128)(flat1)
drop1 = Dropout(0.5)(fc1)
main_output = Dense(1)(drop1)
# -- end architecture --


# sgd parameters
model_path ='./model.h5'
sgd_select = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

car_model = Model(input = [main_input], output = [main_output])
car_model.compile(optimizer = sgd_select, loss = 'mse')

# set data generator
datagen = ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05)

# train network
#car_model.fit(x_data, np.array(list_y), validation_split=0.1, batch_size=16, nb_epoch=6)
#car_model.fit_generator(datagen.flow(x_data[:10000], np.array(list_y[:10000]), shuffle=True, batch_size=16), samples_per_epoch=x_data.shape[0], nb_epoch=7)
car_model.fit_generator(train_generator, samples_per_epoch=15000, validation_data=validation_generator, nb_val_samples=6000, nb_epoch=8)
car_model.save(model_path)
