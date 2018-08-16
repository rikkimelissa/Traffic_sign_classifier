# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Pipeline description

My pipeline consisted of 5 steps.
#### 1. Convert the image to grayscale.
#### 2. Apply Gaussian smoothing.
#### 3. Detect edges with Canny edge detection
#### 4. Choose a quadrilateral region of interest and apply a mask
#### 5. Detect lines with a Hough transform. I used a high threshold, low minimum line length, and high line gap to isolate lane lines.

In order to draw a single line, I separated the lines given by the Hough transform by slope. Anything with a slope outside of a defined region was thrown out. The remaining lines were used to calculate an average slope and y-intercept for each side. I used these averages to define a line in the region of interest.

### Potential shortcomings

This detection does not work well on curves, because the Hough transform uses a high threshold of intersections for something to count as a line, and a curve is not a line!
This detection also doesn't work well with shadows, and it probably also would not work well in bad lighting conditions or bad weather. It would also likely get confused if the road was banked or if there was a high gradient.

### Possible improvements

This pipeline would be more robust if curves were considered in addition to lines. It could also be improved by comparing output between frames, instead of treating each frame as an individual image to analyze. I could be more confident in a line prediction if it was similar to the line detected in the previous image.
