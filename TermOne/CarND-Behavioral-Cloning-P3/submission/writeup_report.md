
## Behavioral Cloning

### Write up document for Behavioral Cloning Project

[//]: # (Image References)
[original_image]: ./images/original_image.jpg "Original Image"

### Goal and Objective

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model. This file is the main pipeline running Nvidia's model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.pdf summarizing the results

### 2. Submission includes functional code

#### Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used to train is Nvidia's self driving car architecture model. The network consists of 9 layers
- 1 normalization layer
- 5 Convolutional layers
- 3 fully connected layers

The model includes RELU layers to introduce non linearity and data is normalized using a Keras lambda layer

Layer  | Description | Output Shape
  ------------- | -------------  | -------------
 Resize Input image  | Sample data pass to model                             |  160 x 320 x 3
 Cropping Image  | Cropped Sample data pass to model                         |  65 x 320 x 3 
 Convolutional 2D  | First Convolution Layer, 2 x 2 stride , relu activation |  31 x 159 x 24
 Spatial Dropout  |        Reduce overfitting            |  31 x 159 x 24
 Convolutional 2D  | Second Convolution Layer, 2 x 2 stride, relu activation  |  14 x 78 x 36
 Spatial Dropout  |        Reduce overfitting            |  14 x 78 x 36
 Convolutional 2D  | Third Convolution Layer, 2 x 2 stride, relu activation  |  6 x 38 x 48
 Spatial Dropout  |       Reduce overfitting             |   6 x 38 x 48
  Convolutional 2D  | Fourth Convolution Layer, relu activation  | 4 x 36 x 64
  Spatial Dropout  |       Reduce overfitting             |  4 x 36 x 64
  Convolutional 2D  | Fifth Convolution Layer, relu activation  | 2 x 34 x 64
  Spatial Dropout  |     Reduce overfitting               |  2 x 34 x 64
  Flatten  |                    |  1 x 4352
  Dropout  |       Reduce overfitting             |  1 x 4352
  Dense 1  |                    |  1 x 100
  Dense 2  |                    |  1 x 50
  Dense 3  |                    |  1 x 10
  Dropout  |       Reduce overfitting             |  1 x 10
  Dense 4  |                    |  1 x 1

#### 2. Attempts to reduce overfitting in the model


The model contains dropout layers in order to reduce overfitting

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

The data set given was used for training.

```
data_given/
|
|-- IMG/
|
|-- driving_log.csv
```

IMG directory has the central, right and left frame of the driving data. Each row in the driving_log.csv sheet corresponds to the images with the steering angle, throttle, brake and speed of the car.


Training data provided by Udacity was split to a ratio of 80:20. 80% for training and 20% for validation.


### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement the LeNet CNN and Nvidia's CNN.

After a few rounds of test, Nvidia's CNN seems more suitable as it is able to navigate track 1 better.I thought this model might be appropriate because it is well documented and discussed by the tutorial video.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding layers of dropout. Dropout can be found in between convolutional 2d layers step.

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Please refer to Model Architecture and Training Strategy section

#### 3. Creation of Training Set & Training Process

The dateset used was given by Udacity.


- The images was cropped top 70 pixels and bottom 25 pixels as suggested by the tutorial in the video
- Images were also augemented by flipping it 180 degrees
- Augmented measurements were also added by negating the values.

This increasees the training dataset

Original Image

![Original Image][original_image]



```python

```
