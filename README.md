# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./bar.png "Bar Graph"
[image2]: ./30kph.jpg "Traffic Sign 1"
[image3]: ./doublecurve.jpg "Traffic Sign 2"
[image4]: ./nopassing.jpg "Traffic Sign 3"
[image5]: ./slippery.jpg "Traffic Sign 4"
[image6]: ./stop.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! The code for my project is in the Udacity workspace as Traffic\_Sign_Classifier and as an html with the same name.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799 images
* The size of the validation set is: 4410 images
* The size of test set is: 12630 images
* The shape of a traffic sign image is: 32x32 pixels
* The number of unique classes/labels in the data set is: 43 classes

#### 2. Include an exploratory visualization of the dataset.

To visually explore the data set, I looked at comparisons of color vs grey and normalizing the images based on color vs gray. Below is a bar chart showing the percentage of classes in each image set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

To preprocess the data, I converted the images to grayscale by taking the average of the three color layers for each image. Then I normalized the images by subtracting 128 and dividing by 128. I chose these techniques because they are what was used in the paper by Sermanet and LeCun for their ConvNet classifier. I did not generate additional data because this method worked well enough.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Fully connected		| outputs 120 vector							|
| RELU					|												|
| Fully connected		| outputs 84 vector 							|
| RELU					|												|
| Dropout					|Probability .5									|
| Fully connected		| outputs 43 classes 							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used mostly the same approach implemented in the LeNet example. I used the softmax of the cross entropy of one hot encoding the class labels. I used the Adam Optimizer to minimize the loss based on the mean of the cross entropy softmax. I trained for 15 epochs, and I reduced the learning rate as the epoch number increased. Batch size I kept at 128. I initialized the weights and biases with a truncated normal distribution centered at zero and a standard deviation of .1.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of .996
* validation set accuracy of .947
* test set accuracy of .92
These results were calculated in the cell right after the training.

My approach to finding a solution was iterative. I started with the LeNet model, and made modifications using approaches suggested in the lessons and based on the paper by Sermanet and LeCun in the following order: 
* The original LeNet approach only gave me a validation accuracy of .687.
* Based off the paper, I converted the images to grayscale and got a validatin accuracy of .908.
* As a simple test, I increased the learning rate to .002 and increased the standard deviation in the weight and bias initialization, giving a validation accuracy of .89.
* I then started to change the model structure by taking out of the fully connected layers at the end but increasing the depth of the convolution layers, giving a validation accuracy of .9, based on the paper conv net structure.
* This model change didn't help much, so I changed the structure back to LeNet but with only two fully connected layers, and added dropout with a keep probability of .7. This increased validation accuracy a little but not enough.
* Lastly, I increased the number of epochs to 15, added the third fully connected layer back in, and reduced dropout to .5. This time I remembered to evaluate the validation accuracy with zero dropout and got an accuracy of .94.

Throughout this process I was not comparing the training accuracy to the validation accuracy, so I can't comment on over or under fitting. One thing I noticed was that the validation accuracy would sometimes decrease as the epochs went on, which is the reason I added dropout and decided to lower the learning rate as the epochs progressed. I think dropout really helped make my model more robust.

The final accuracy numbers provide evidence that the model is working well on the provided images.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The third and fourth image might be hard to classify because they get really pixelated after being resized. The second image might be hard to classify because there are not many examples of it in the test set.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30) 		| Speed Limit (30)								| 
| Double curve    		| Dangerous right curve 						|
| No passing			| No passing									|
| Slippery road	  		| Priority road					 				|
| No entry  			| No entry    		 							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is well below the test set accuracy of 92%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is really sure that this is a 30kmh sign, with a probability of almost 100%. The four other maximum value signs all have a probability of less than 10^-21 and are 50kmh, 70kmh, 20kmh, and wild animal crossing.

For the second image, the model classifies the sign as a dangerous right curve with a probability of 97%, and to be fair to the model these signs are pretty similar. The four other signs are slippery road, children crossing, bicycle crossing, and wild animal crossing with 1.4%, 1.3%, <.05%, <.05% respectively. All of these signs are triangular with red borders, but the correct sign does not show up at all here.

For the third image, the model classified the sign correctly as no passing with a probability of 98%. The other probabilities are all less than 1% and are slippery road, dangerous left curve, no passing for heavy vehicles, and right turn ahead.

For the fourth image, the model incorrectly classifies the sign as a priority road with a probabibility of 99%. The other probabilities are all less than 1% for right of way, right turn ahead, ice/snow, and road work. The correct sign of slippery road appears nowhere. The top two choices for this image also look nothing alike or like the real image. This could probably be improved with more test images of this type, but I'm not sure why the neural net is so bad at classifying this type of sign.

For the fifth image, the model correctly classified the sign correctly as no entry with a probability of almost 100%. The other probabilities are less than 10^-9 and are stop, 80kmh, right turn ahead, and right of way.

These probabilities show that regardless of being correct or incorrect, my model is always very sure of itself. A way to fix this might be to add in distorted images or fake data as suggested in this project.

