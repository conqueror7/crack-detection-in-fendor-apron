'''
Before running this augmentation code make sure you have created the replica of your original folder and pass it as a destination
'''
import os
import sys
import glob
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


'''
sys.argv[1] original folder 
sys.argv[2] destination folder (replica of original folder)
sys.argv[3] # of images to be generated in each folder 
'''
rootdir = sys.argv[1]  

datagen = ImageDataGenerator(
        rotation_range=20,
        #width_shift_range=0.20,
        #height_shift_range=0.15,
        #shear_range=0.10,
        #zoom_range=0.10,
        horizontal_flip=True,
        vertical_flip=True,
        zca_whitening=True
		#zca_epsilon=1e-6,		
        #fill_mode='nearest')
	
for objectDirectory in next(os.walk(rootdir))[1]:
	saveToDir=sys.argv[2]+"/"+objectDirectory
	#os.makedirs(saveToDir)
	
	no_of_images=len(glob.glob(os.path.join(rootdir,objectDirectory, "*")))
	image_factor=int(sys.argv[3])/int(no_of_images)
	print '# images in '+objectDirectory+' : '+str(no_of_images)
	print 'factor : '+ str(image_factor)
		
	for image in glob.glob(os.path.join(rootdir,objectDirectory,"*")):
		img = misc.imread(image) # this is a PIL image
		x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
		x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
		#no_of_images_dir=len(glob.glob(os.path.join(saveToDir, "*.jpg")))
		
		
		# the .flow() command below generates batches of randomly transformed images
		i = 0
		for batch in datagen.flow(x, batch_size=1,save_to_dir=saveToDir, save_prefix=objectDirectory, save_format='jpg'):
			i += 1	
			if i > image_factor:
				break  # otherwise the generator would loop indefinitely
			if len(glob.glob(os.path.join(saveToDir, "*"))) >=int(sys.argv[3]):
				print 'limit reached'
				print len(glob.glob(os.path.join(saveToDir, "*")))
				break
		
		if len(glob.glob(os.path.join(saveToDir, "*"))) >=int(sys.argv[3]):
			print 'limit reached'
			print len(glob.glob(os.path.join(saveToDir, "*")))
			break
