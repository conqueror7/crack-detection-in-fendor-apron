#load the required libararies 

import glob
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import sys
#load the keras libraries

from keras.layers import Dropout, Input, Dense, Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint


def getClassNames(validation_path):
	file_names = glob.glob(validation_path + '\\*\\*.*')
	original_label = []
	for file in file_names:
	    original_label.append(file.split('\\')[-2])
	lb = LabelBinarizer().fit(original_label)
	label = lb.transform(original_label	) 

	return lb

def loadData(test_data_path,img_width, img_height):
	z = glob.glob(test_data_path + '\\*.*')
	test_imgs = []
	names = []
	for fn in z:
	    names.append(fn.split('\\')[-1])
	    new_img = Image.open(fn)
	    test_img = ImageOps.fit(new_img, (img_width, img_height), Image.ANTIALIAS).convert('RGB')
	    test_imgs.append(test_img)

	return names,test_imgs

'''
sys.argv[1] validation folder 
sys.argv[2] test folder
sys.argv[3] model path
sys.argv[4] image width
sys.argv[5] image height 
'''

if __name__ == '__main__':

	validation_path = sys.argv[1] 
	test_data_path =  sys.argv[2]
	model_path = sys.argv[3]
	img_width = int(sys.argv[4])
	img_height = int(sys.argv[5])
	print(validation_path)
	print(test_data_path)
	print(model_path)
	print(img_width,img_height)
	#get LabelBinarizer for getting class_names
	lb = getClassNames(validation_path)

	#load test data into memory as we have only 5 images, if there are a lot of images we can do this batch-wise

	file_names,test_images = loadData(test_data_path,img_width, img_height)

	#load the saved model
	model = load_model(model_path)

	test_imgs = np.array([np.array(image) for image in test_images])
	test_imgs = test_imgs.astype(float)
	test_data = test_imgs.reshape(test_imgs.shape[0], img_width, img_height, 3) / 255

	#predict 

	predictions = model.predict(test_data)
	confidence = []
	for prediction in predictions:
		confidence.append(max(prediction))

	predicted_classes = lb.inverse_transform(predictions)

	#create submission file 

	df = pd.DataFrame(data={'file': file_names, 'class': predicted_classes,'confidence' : confidence})
	df_sort = df.sort_values(by=['file'])
	df_sort.to_csv('results.csv', index=False)
