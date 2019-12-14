import os
import tensorflow as tf
import keras
import time
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger


DATASET_PATH  = '/deepLearning/01/data/pneumonia/chest_xray'
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = 2
BATCH_SIZE    = 200
NUM_EPOCHS    = 100
WEIGHTS_FINAL = 'model-pneu01-final-' + str(int(time.time())) + '.h5'


train_datagen = ImageDataGenerator( rescale=1.0/255.0,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
								   
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/test',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64,(3,3)))
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128,(3,3)))
model.add(LeakyReLU(alpha=.001))
model.add(Flatten())
model.add(Dense(units=32,activation='sigmoid'))
model.add(Dense(units=2,activation='softmax'))

adam = Adam(lr=.001)

model.compile(optimizer = adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

csv_logger = CSVLogger(filename="log.csv"+str(int(time.time())))
act_filepath = './weights' + str(int(time.time()))
os.mkdir(filepath)
save_weights = ModelCheckpoint(filepath = act_filepath + '/weights.{epoch:02d}-{val_acc:.2f}.hdf5',monitor='val_acc', verbose=1, save_best_only=True, mode='max')
cblist = [csv_logger, save_weights]

# train the model
model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
			callbacks=cblist)

# save trained weights
model.save(WEIGHTS_FINAL)

