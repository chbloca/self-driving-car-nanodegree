# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: train%20dataset%20distribution.png "Visualization"
[image4]: my_images/example_1.jpg "Traffic Sign 1"
[image5]: my_images/example_6.jpg "Traffic Sign 2"
[image6]: my_images/example_7.jpg "Traffic Sign 3"
[image7]: my_images/example_9.jpg "Traffic Sign 4"
[image8]: my_images/example_8.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data looks like this:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
As a first step, I decided to convert the images to grayscale because color features are not relevant enough to the classifier and also in order to not misclassify images under poor light conditions (poor light conditions usually leads images to have poor color features).
Secondly, a normalization is applied in order to make the data mean zero and equal variance. This is proven to help during the learning process.
No additional data has been generated.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer | Description	| 
|:---------------------:|:---------------------------------------------:| 
| Input | 32x32x1 Gray image | 
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU |					|
| Max pooling | 2x2 stride, outputs 14x14x6	|
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU |					|
| Max pooling | 2x2 stride, outputs 5x5x16 |
| Flatten | outputs 400 |
| Fully connected	| outputs 120 |
| RELU |					|
| Fully connected	| outputs 84 |
| RELU |					|
| Fully connected	| outputs 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 100 epochs, a batch size of 128 and a learning rate of 0.001.
The training optimizers that have been used for the training were:
* softmax_cross_entropy_with_logits: provides a tensor representing the mean loss value. Afterwards, to this is aplied reduce_mean, so that the mean of the elements across dimensions is retrieved
* AdamOptimizer: it is an enhanced version of the SGD and it is used to update network weights iteratively.

My final model Validation Accuracy was 0.963

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.963
* test set accuracy of 0.945

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Standard LeNet
* What were some problems with the initial architecture?

Not substancial ones

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

It required of Dropout applications on the fully connected layers so that the model would not easily overfit.

* Which parameters were tuned? How were they adjusted and why?

Epochs, batch size and keep_prob for dropouts.

If a well known architecture was chosen:
* Why did you believe it would be relevant to the traffic sign application?

Since it was initially designed to perform over the MNIST data, to classify characters. This feature might also be extrapolable to traffic signs. It turns out it does quite a proper job.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The accuracy in training is a metric that indicates if the network is performing well on the forward and back propagation.

If the accuracy in validation is not adequate and the training accuracy is good, it is a symptom of overfitting.
If the accuracy in validation is neither adequate and the training accuracy is not good, it is a symptom of underfitting.

The accuracy in testing indicates the evidence that the model can generalize what has learned to new data.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The provided images are made under good light and shooting conditions (no darkness, blurry, with artificial distortions, ...) so that the classifier most likely will performe quite well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the predictions:

| Image	label | Prediction label | 
|:---------------------:|:---------------------------------------------:| 
| 31 | 31 |
| 12 | 12 | 
| 13 | 13 | 
| 34 | 34 | 
| 17 | 17 | 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.
For instance, for the 4th sample, the top five soft max probabilities were

| Probability | Prediction label | 
|:---------------------:|:---------------------------------------------:| 
| 99.9802649021 % | 34 | 
| 0.019738310948 % | 38 |
| 2.06127359625e-07 % | 17 |
| 5.48926613257e-12 % | 13	|
| 4.02001802886e-12 % | 14	|

This means that the model was extremely confident about the prediction.
