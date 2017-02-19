import cv2
import csv
import numpy as np
import random
import pandas
import sklearn

from random import shuffle

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Activation, Input, Embedding, Lambda, ELU, LSTM, Dense, merge, Convolution2D, MaxPooling2D, Reshape, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from keras import initializations
from keras.callbacks import ModelCheckpoint

from sys import getsizeof
# beneficios novia paisa


# function to filter images according to a histogram using angles
def filter_samples(list_im_angles, num_max_per_sample = 1200):
    filter_list = []

    all_angles = []
    for sample_line in list_im_angles:
        all_angles.append(float(sample_line[3]))
    all_angles = np.array(all_angles)


    bin_angles = np.arange(-25.05,25.1,0.1)
    hist, label = np.histogram(all_angles, bins=bin_angles)
    inds = np.digitize(all_angles, bin_angles)
    for k_bin in range(bin_angles.shape[0]):
        list_filter_bin = [list_im_angles[k] for k in range(len(inds)) if inds[k]==k_bin]
        if len(list_filter_bin) > num_max_per_sample:
            list_filter_bin = random.sample(list_filter_bin, num_max_per_sample)
        filter_list = filter_list + list_filter_bin
    return filter_list


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
        for offset in range(0, int(np.floor(num_samples/batch_size)), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if len(batch_sample[0].split('/')) > 3:
                    name = batch_sample[0]
                    rname = batch_sample[1]
                    lname = batch_sample[2]
                else:
                    name = './IMG/' + batch_sample[0].split('/')[-1]
                    lname = './IMG/' + batch_sample[1].split('/')[-1]
                    rname = './IMG/' + batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name)
                left_image = cv2.imread(lname) 
                right_image = cv2.imread(rname)
                if batch_sample[3] != "steering":
                    center_angle = float(batch_sample[3])
                    left_angle = center_angle 

                    # data augmentation
                    center_image = center_image[20:155]
                    center_image = cv2.resize(center_image, (320,160))
                    #center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    center_image = center_image.astype(np.float32)
                    
                    left_image = left_image[20:155]
                    left_image = cv2.resize(left_image, (320,160))
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    left_image = left_image.astype(np.float32)

                    right_image = right_image[20:155]
                    right_image = cv2.resize(right_image, (320,160))
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    right_image = right_image.astype(np.float32)
                    
                    flipped_center_image = np.copy(center_image)
                    flipped_center_image = np.fliplr(flipped_center_image)

                    images.append(center_image)
                    angles.append(center_angle)
                    
                    #images.append(left_image)
                    #angles.append(center_angle + offset_angle)
                    #images.append(right_image)
                    #angles.append(center_angle - offset_angle)
                    
                    images.append(flipped_center_image)
                    angles.append(-1*center_angle)
                    # -- end data augmentation --

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# read csv file
samples = []
list_log = ['/home/darial/udacar/beta_simulator_linux/driving_log.csv']
for log_file in list_log:
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

flag_filter = True
if flag_filter:
    samples = filter_samples(samples)
    print("images filtered")
print("total images", len(samples))
    
# split data train and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create generators
num_batch = 16
train_generator = generator_batch_images(train_samples, batch_size=num_batch)
validation_generator = generator_batch_images(validation_samples, batch_size=num_batch)


# --first architecture--
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

# second architecture
""" from: https://github.com/commaai/research/blob/master/train_steering_model.py  """
row, col, ch = 160, 320, 3
L2_REG_SCALE = 0.
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row, col, ch),output_shape=(row, col, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', W_regularizer=l2(L2_REG_SCALE)))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512, W_regularizer=l2(0.)))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1, W_regularizer=l2(0.)))
# -- end second architecture ---

# sgd parameters
model_path ='./model.h5'
#sgd_select = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

car_model = Model(input = [main_input], output = [main_output])
#car_model.compile(optimizer = sgd_select, loss = 'mse')
model.compile(optimizer = 'Adam', loss = 'mse')

# checkpoint to save
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print("number parameters original", car_model.count_params())
print("number parameters commaai", model.count_params())

# train network
#car_model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples), validation_data=validation_generator, nb_val_samples=2*len(validation_samples), nb_epoch=10)
samples_epoch_train = 2*num_batch*(np.floor(len(train_samples)/num_batch))
samples_epoch_val =  2*num_batch*(np.floor(len(validation_samples)/num_batch))
model.fit_generator(train_generator, samples_per_epoch=samples_epoch_train, validation_data=validation_generator, nb_val_samples=samples_epoch_val, callbacks=callbacks_list, nb_epoch=11)
model.save(model_path)
