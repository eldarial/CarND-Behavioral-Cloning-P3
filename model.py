import cv2
import numpy as np
import pandas

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


# function to initialize layer weights in network
def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

#list_of_log_files = ['driving_log.csv', '/home/darial/udacar/CarND-Behavioral-Cloning-P3/simulator-linux/driving_log.csv']
list_of_log_files = ['/home/darial/udacar/beta_simulator_linux/driving_log.csv']
list_x_path, list_y =  get_data_list(list_of_log_files)
x_data = get_batch_images(list_x_path)
print("max",np.max(x_data),"min",np.min(x_data),x_data.shape,len(list_y))

# --create architecture--

main_input = Input(shape=(66, 200, 3), name='main_input')

conv1 = Convolution2D(16, 5, 5, border_mode='valid')(main_input)
conv2 = Convolution2D(24, 5, 5, border_mode='valid')(conv1)

conv3 = Convolution2D(24, 5, 5, border_mode='valid')(conv2)
conv4 = Convolution2D(48, 3, 3, border_mode='valid')(conv3)
conv5 = Convolution2D(32, 3, 3, border_mode='valid')(conv4)

flat1 = Flatten()(conv5)

fc1 = Dense(64)(flat1)
drop1 = Dropout(0.5)(fc1)
main_output = Dense(1)(drop1)

# -- end architecture --

# sgd parameters
model_path ='./model.h5'
sgd_select = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


car_model = Model(input = [main_input], output = [main_output])
car_model.compile(optimizer = sgd_select, loss = 'mean_squared_error', metrics=['mean_squared_error'])

# set data generator
datagen = ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05)

# train network
car_model.fit(x_data, np.array(list_y), validation_split=0.1, batch_size=16, nb_epoch=6)
#car_model.fit_generator(datagen.flow(x_data[:10000], np.array(list_y[:10000]), shuffle=True, batch_size=16), samples_per_epoch=x_data.shape[0], nb_epoch=7)
car_model.save(model_path)
