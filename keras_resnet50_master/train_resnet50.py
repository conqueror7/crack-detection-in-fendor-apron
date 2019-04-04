##Importing Libraries

import pandas as pd
import numpy as np
import os
import random
import shutil
import math
import sys
import glob
from time import time
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
from scipy import misc

# learning rate schedule
def step_decay(EPOCH):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+EPOCH)/epochs_drop))
    print("Epoch: {0:} and Learning Rate: {1:} ".format(EPOCH, lrate))
    return lrate

if __name__ == '__main__':
	## Setting Global variables
	DATA_TRAIN=sys.argv[1]
	DATA_VALID=sys.argv[2]
	save_loc=sys.argv[3]
	BATCH_SIZE=sys.argv[4]
	EPOCH = sys.argv[5]
	num_classes = len(os.listdir(DATA_TRAIN))
	nb_train_samples = sum([len(files) for r, d, files in os.walk(DATA_TRAIN)])
	nb_validation_samples = sum([len(files) for r, d, files in os.walk(DATA_VALID)])

	print("Training Samples :",nb_train_samples)
	print("Validation Samples :",nb_validation_samples)
	print("Number of Classes :",num_classes)
	
	#we can use any model from the below list of image classification model to train the network
	model_name='resnet50'

	##Initialising Resnet50 model taking only convolution layers. Input size to the model is 224 x 224. 
	img_width, img_height = 224, 224
	model=applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
	model.layers.pop()

	##Adding fully-connected layers on top of resnet convolution network. All layers including convolution layers in the model will be trained. 
	for layer in model.layers:
		layer.trainable = True

	pretrained_inputs = model.inputs
	model = Flatten()(model.output)
	model = Dense(512,activation='relu')(model)
	model = Dropout(0.5)(model)
	predictions = Dense(num_classes,activation='softmax')(model)

	model_final = Model(inputs=pretrained_inputs, outputs=predictions)

	##If model weights are already present, those weights can be loaded to the network.
	# load weights
	#model_final.load_weights("models\\resnet50 weights-best.hdf5")

	##Compiling the model using loss funtion of categorical crossentropy
	# Compile the model
	model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

	##Initialising Data Generators for Batch-wise training
	# Initiate the train and test generators with data Augumentation
	train_datagen = ImageDataGenerator(
		rescale = 1./255)
	
	test_datagen = ImageDataGenerator(
		rescale = 1./255)
	
	train_generator = train_datagen.flow_from_directory(
		DATA_TRAIN,
		target_size = (img_height, img_width),
		batch_size = BATCH_SIZE,
		class_mode = "categorical")

	validation_generator = test_datagen.flow_from_directory(
		DATA_VALID,
		batch_size = BATCH_SIZE,
		target_size = (img_height, img_width),
		class_mode = "categorical")

	##Saving the model if any improvement in validation accuracy
	model_save = save_loc + '\\'+model_name +"-weights-best.hdf5"
	checkpoint = ModelCheckpoint(model_save, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, period=1,mode='max')
	early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=35, verbose=1, mode='auto')

	change_lr = LearningRateScheduler(step_decay)

	# Train the model (Experiment2)
	model_final.fit_generator(
		train_generator,
		samples_per_epoch = nb_train_samples,
		epochs = EPOCH,
		validation_data = validation_generator,
		validation_steps = nb_validation_samples,
		callbacks = [checkpoint, early_stopping, change_lr])

	# Save the final model on the disk
	model_final_name = save_loc +'\\'+ model_name +"-weights-final.hdf5"
	if not os.path.exists(os.path.dirname(model_final_name)):
		try:
			os.makedirs(os.path.dirname(model_final_name))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	model_final.save(model_final_name)

