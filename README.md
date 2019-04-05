# crack-detection-in-fendor-apron

Classifying images of fendor apron into healthy and defected class using object detection algorithm- Faster R-CNN.

#### Setup info :
* Keras : 2.2.4
* Tensorflow : 1.13.1

#### Overview of dataset
There are 200 total images in the dataset. We have two classes :
* Healthy : 111 images
* Defected : 89 images <br />
Image dimensions :
* Image Type : jpg(JPEG)
* Width x Height : 3120 x 4160
* Images are rescaled to Width x Height : 640 x 480

Example of 'Healthy' class :
![healthy](https://user-images.githubusercontent.com/24800950/55561614-1383aa80-5710-11e9-9ca2-15e55f264c73.jpg)
Example of 'Defected' class :
![defected](https://user-images.githubusercontent.com/24800950/55561608-11b9e700-5710-11e9-9820-33bf81f16c83.jpg)

#### Classification Approach
Treated it as binary classification problem and followed below steps-
* Increased the data size from 200 to 600 after augmenting images of each class, healthy(300) and defected(300). Parameters used for augmentation are horizontal flip, vertical flip, rotation by 90 degree angle. 
* Used Resnet50 as base classication model and build Fully connected network over it.
* Trained the whole network (convolution and fully connected layers) for updating weights. 

#### Detection Approach
Treated it as one class object detection problem. Trying to identify defect in images using object detection algorithm like Faster R-CNN. Images with no defect will be classified as healthy and with defect as defected.
Steps followed-
* Took 300 augmented defected images and annotated the data using Labelbox tool (https://labelbox.com/).<br />
Created bounding box around the cracked/defected portion of object. Check following examples:
![defected_1](https://user-images.githubusercontent.com/24800950/55561609-12527d80-5710-11e9-9b65-8341d3039a5f.JPG)

![defected_2](https://user-images.githubusercontent.com/24800950/55561611-12eb1400-5710-11e9-995a-d5af1d40be0a.JPG)

* Trained Faster R-CNN model with VGG conv-net on the defected images.
* The anchor box sizes used are [64, 128, 256, 512] and the ratios are [1:1, 1:2, 2:1] <br />


#### Steps to run the code
**Resnet50 model**
* For training Resnet50 model, use the command below <br />
`python train_resnet50.py train_dir_path validation_dir_path  model_save_path batch_size num_epochs`

* To predict using Resnet50 model, use the command below <br />
`python test_resnet50.py validation_dir_path test_dir_path model_name 224 224`

**Faster R-CNN model**
* For training Faster R-CNN model, use the command below <br />
`python train_frcnn.py -p annotate_aug_scaled.txt -o simple --network vgg --num_epochs 100`

* To predict using Faster R-CNN model, use the command below <br />
`python test_frcnn.py -p test_folder_path` <br />

**Utils Folder**
* Augmentation can be done using ImageDataGenerator class in keras. Run below command <br />
`python image_augmentation.py original_folder destination_folder num_images_to_be_generated`
* img_rescale_util.ipynb can be used to resize the image size if too big and can help in saving training time.
* annotations_json_to_csv_util.ipynb can be used to parse the json file obtained after annotating the images in Labelbox for detection. Output file will be .txt file in below format <br />
`img_path,x_min,x_max,y_min,y_max,class` <br />
This annotate_aug_scaled.txt file will be passed as input for Faster R-CNN model training.
