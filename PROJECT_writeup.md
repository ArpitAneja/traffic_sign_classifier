# **Traffic Sign Recognition** 

## Writeup
[image1]: ./my_images/random/1.png
[image2]: ./my_images/random/2.png
[image3]: ./my_images/random/3.png
[image4]: ./my_images/random/4.png
[image5]: ./my_images/random/5.png
[image6]: ./my_images/bar/1.png
[image7]: ./my_images/bar/2.png
[image8]: ./my_images/bar/3.png
[image9]: ./new_images/0.jpg
[image10]: ./new_images/14.jpg
[image11]: ./new_images/18.jpg
[image12]: ./new_images/25.jpg
[image13]: ./new_images/27.jpg
[image19]: ./new_images/3.jpg
[image14]: ./my_images/prediction/1.png
[image15]: ./my_images/prediction/2.png
[image16]: ./my_images/prediction/3.png
[image17]: ./my_images/prediction/4.png
[image18]: ./my_images/prediction/5.png
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
---

## Step 0: Load The Data
The data(training,validation and test) is loaded from the pickle file provided.

---

## Step 1: Dataset Summary & Exploration
 
 1. Providing a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
    I used the numpy library to calculate summary statistics of the traffic signs data set: The number of training examples,testing examples,image data shape and number of classes are listed below.
    
    *The size of training set is 34799 samples
    *The size of the validation set is 4410 samples
    *The size of test set is 12630 samples
    *The shape of a traffic sign image is (34799, 32, 32, 3)
    *The number of unique classes/labels in the data set is 43

 2. Including an exploratory visualization of the dataset
    Here i have created a function to get the Sign name from the classID.
    pandas is used to read csv file. It takes input as classId and returns the corresponding sign name.
    def getSignName(classId):
       
```
import pandas as pd
sign_names = pd.read_csv("./signnames.csv",names=["ClassId","SignName"])
return sign_names["SignName"][classId]
```
        

  1. Plotting random 5 images from the dataset and displaying the corresponding Sign name with it.
        
        ![alt text][image1]
        ![alt text][image2]
        ![alt text][image3]
        ![alt text][image4]
        ![alt text][image5]
    
   2. plotted the bar graph for number points per class vs class to analyse the datasets
        
        ![alt text][image6]
        ![alt text][image7]
        ![alt text][image8]

        by analysing the graph i observed two things
         
         1. Some classes are 10 times of other classes, hence the classes are very imbalanced in all three, train, test and validation, datasets. 

        2. all three dataset's classes are distributed in similar fashion 

## Step 2: Design and Test a Model Architecture
 1. In this step i will describe what preprocessing steps i used and what model i choosed for this project.
    1. Pre-processing the data
        *The data was from different scales so adjustment of values to a comman scale i.e normalization was required for all three train, test and validation, data sets.
        *i used min-max scaling for normalization
          * it brings the data between 0 to 1 scale
          * linearly transforms x to y= (x-min)/(max-min)
          * it further speeds up training procedure
  2. Model Architecture

       * I used LeNet model given in the CNN module of this course.
       * here we have 5 layers.
        ---
       * used following in few layers:
            * RELU : 
                 used it in first four layers.
                 rectified linear units. 
                 it is a type of activation function.
                 f(x) = max(0,x) , returns 0 if -ve and returns x if +ve

            * Dropout:
                 used it in first two layers
                 drops some units from network
                 reduce over-fitting
                 it is used to improve performance of network so that our network do not relies on particular activations
                 here network is forced to learn redundant representation.

            * Pooling:
                 used max pooling in first two layers
                 takes all convolutions in neighbourhood and combines them to reduce size of input so that focus should be on important data only.
                 y = max(Xi)

            * Softmax:
                 sigmoid function used as output function for last layer


       * following is the detailed view of layers used in architecture of my model
       
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x6   | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Dropout	      	    | With keep probability as 0.8 during training 	|
| Convolution 5x5x6x16	| 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Dropout	      	    | With keep probability as 0.8 during training 	|
| Fully connected		| Flatten layer with 400 input and 120 output   |
| RELU					|												|
| Fully connected		| Flatten layer with 120 input and 84 output    |
| RELU					|												|
| Fully connected		| Flatten layer with 84 input and 43 output     |
| Softmax				|         									    |


## Train, Validate and Test the Model

How i trained the model:

* The training data is shuffled before passing to the model. 
* I have used an **AdamOptimizer** because a learning rate is maintained for each network weight (parameter) with **batch size = 128**, **epochs = 40** and **learning rate = 0.001** to train the model.
* I used following methods to train the model and to reach to the final model.

    1. First I used training set directly which was giving around 0.8 accuracy on validation set but around 0.87 accuracy on training set. 

    2. Then using Min-Max scaling  i normalized the data which improved the validation set accuracy till 0.89.

    3. I observed model was overfitting for the training set and validation set accuracies.

    4. To remove ovefitting,i used **dropout** and chose **keep_prob = 0.8** for training and **keep_prob = 1** for calculating the accuracies.

    6. **Dropout** method introduces **dropout layer** in each convolution layer after **maxpooling** layer.

    7. FInally I was able to get following accuracies on all three datasets.

        
        **Train Accuracy = 1.000**

        **Validation Accuracy = 0.952**

        **Test Accuracy = 0.947**

-----------

## Step 3: Test a Model on New Images

##### Load and Output the Images

1. i downloaded few images from web and named them according to thier class id

2. following are the images
    ![alt text][image9]
    ![alt text][image10]
    ![alt text][image11]
    ![alt text][image12]
    ![alt text][image13]
    ![alt text][image19]

3. Pre-processing is done on the newly downloaded images.That is all images were normalized by using Min-Max scaling.
   
    ```
    X_new_image_p, y_new_image_p = preprocess(X_new_image, y_new_image)
    ```
    
##### Predict the Sign Type for Each Image

###### Analyze Performance

* Then accuracy was measured by using saved model and model could easily recognize the all 5 images.
* the model predicted all the images correctly
* i got the accuracy of the model with new image as follows:
    **New Image Accuracy = 1.000**
* Softmax probablities were calculated by using logits for the new images
* Given below is the graphs of new images showing correct predicted class with much much higher value than the other classes:
   
   ![alt text][image14]
   ![alt text][image15]
   ![alt text][image16]
   ![alt text][image17]
   ![alt text][image18]
