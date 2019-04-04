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

Example of classes :
![healthy](./healthy.jpg)
![defected](./defected.jpg)

#### Classification Approach
Treated it as binary classification problem and followed below steps-
* Increased the data size from 200 to 600 after augmenting images of each class, healthy(300) and defected(300). Parameters used for augmentation are horizontal flip, vertical flip, rotation by 90 degree angle. 
* Used Resnet50 as base classication model and build Fully connected network over it.
* Trained the whole network (convolution and fully connected layers) for updating weights.
Results
* Training Accuracy :
* Validation accuracy : 
* Test Accuracy : 

#### Detection Approach
Treated it as one class object detection problem. Trying to identify defect in images using object detection algorithm like Faster R-CNN. Images with no defect will be classified as healthy and with defect as defected.
Steps followed-
* Took 300 augmented defected images and annotated the data using Labelbox tool. Created bounding box around the cracked portion of object. Check following examples-
![defected_1](./defected_1.jpg)
![defected_2](./defected_2.jpg)
* Trained Faster R-CNN model with VGG conv-net on the defected images.
* The anchor box sizes used are [64, 128, 256, 512] and the ratios are [1:1, 1:2, 2:1]
Training results :
* Mean number of bounding boxes from RPN overlapping ground truth boxes: 0.9559068219633944
* Classifier accuracy for bounding boxes from RPN: 0.99996875
* Loss RPN classifier: 0.18826815635299832
* Loss RPN regression: 0.004031765677774274
* Loss Detector classifier: 0.00011033100617267167
* Loss Detector regression: 0.005953244450405691
Test results :
* Accuracy :

#### Steps to run the code
* For training Resnet50 model :

* For testing on Resnet50 model :

* For training Faster R-CNN model :
python train_frcnn.py -p annotate_aug_scaled.txt -o simple

* Inferencing Faster R-CNN model :
python test_frcnn.py -p test_folder_path
