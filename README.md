#Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data. 

## How to execute code

* First you've to download the data set and uncompress it into the data directory. See [README.md](./data/README.md).
* Train the classifier `train.py`. Output is saved into `model.p`.
* Detect cars in an image or video `process.py test_video.mp4`

[//]: # (Image References)
[vehicle]: ./writeup/vehicle.png
[non-vehicle]: ./writeup/non-vehicle.png
[car-detection-1]: ./writeup/windows-1.jpg
[car-detection-2]: ./writeup/windows-2.jpg
[sliding-windows]: ./writeup/sliding-windows.jpg
[detection-1-windows]: ./writeup/detection-1-windows.jpg[detection-2-heatmap]: ./writeup/detection-2-heatmap.jpg[detection-3-heatmap-thresholded]: ./writeup/detection-3-heatmap-thresholded.jpg[detection-4-output]: ./writeup/detection-4-output.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

| ![][vehicle] 	| ![][non-vehicle]	|
|----------------|-------------------|
| Vehicle		  	|Non-vehicle		 	|


HOG features are extracted by `get_hog_features()` in `common.py`. The heavy lifting is done by `skimage.feature.hog()`. This method has a `orientations`, `pixels_per_cell`, and `cells_per_block` parameters. 


####2. Explain how you settled on your final choice of HOG parameters.

I had to do some exploratory work and decided to use my classifier accurary/score to find the HOG arguments value maximizing it.

For our given dataset, `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` applied to all channels provide the best result. Those arguments are defined in `train.py` at the top of the file and are persisted to disk along the trained classifier.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the help of `sklearn.svm.LinearSVC` in `train.py` file. I reserved a random 20% of the dataset as a test set to compute the accuracy of my classifier. The classifier perform well, with an accuracy that is frequently above 0.98.

Unfortunately, the dataset has one weakness, it contains a lot of images that have been extracted from a video stream as quite a few images seem to be similar (very little perspective or size change). This means our training set leaks into our testing set and prevents us to have a robust accuracy. I'm aware of this problem, and I decided to ignore it and focus on other challenges.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented by `find_cars()` in `process.py`.

In my implementation the windows size is fixed. They are better ways to do it with dynamic windows and it will be debated in the discussion section of this writeup.

As my window size is fixed and my classifier has been trained on 64x64 images, I decided to have a window size of 64x64. This means I had to resize the image in such way that my windows size could be a bounding box for cars in the image. I came up with the x1.5 ratio. This means I blew up the input image by 1.5 in order for my window to detect cars. 

Each window has an overlap of 25%, step of 16 pixels. Only plossible area of the image are searched, meaning I only scan the lower bottom of the image. This strategy results in 294 windows per image.

![][sliding-windows]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![Example of car detection][car-detection-1]
![Another example of car detection][car-detection-2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

It's all tacking place in `CarTracker.next_image()` in `process.py`. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Positive detections:

![][detection-1-windows]

Heatmap:

![][detection-2-heatmap]

Thresholded heatmap:
![][detection-3-heatmap-thresholded]

Output:
![][detection-4-output]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline I came up with works relatively well on the project video, but they are a few things I could improve.

I already mentioned that our dataset provides a lot of images that have been extracted from the same video stream and the same tracked vehicle. This means that our training set and test set are to some degree similar, another way to see it, our training set leaks into our test set which in turn means our classifier accuracy could be more robust than it's right now. Ideally, images that are too similar should be in only one of the two sets.

Sliding windows search are performed with a fixed size window. It was enough to get through the project video but it can be problematic as cars in the far distance will appear smaller than cars close to the camera. One approach to fix this would be to search the horizon with a smaller window size and search near distance with a bigger window size.

Another area to explore is the fact that we apply the pipeline to a video. This means we could smooth out the detected boxes and ignore more false positives by filtering out intermittent or sporadic detections. This can be done by blending the search result of x number of images or a specified amount of time. The challenge will be to find the right balance between not delaying to much the detection of new cars appearing and filtering out false positives.  

