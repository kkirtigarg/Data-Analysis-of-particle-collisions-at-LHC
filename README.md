# Data-Analysis-of-particle-collisions-at-Large Hadron Collider

## Project: Exploring Beyond the Standard Model with High-Energy Physics

### Introduction and Motivation
- High-energy physics endeavours to understand fundamental nature through particle collisions at extreme energies.
- LHC experiments (ATLAS, CMS) amassed vast data; the Standard Model explains only 4% of the universe.
- Pursuit of physics models beyond the Standard Model seeks to unravel the unknown phenomena.

### Project Goal
- Investigate precision in the SM via the LHC's potential.
- Discover novel physics and maintain background rejection.
- Effective preprocessing(blurring and Deblurring) of the input images using autograd for accurate classification of signal and background events.
- Employ Machine Learning (ML) and Deep Learning algorithms to enhance signal efficiency.

### Introduction

The pursuit of unravelling the mysteries of the universe has led physicists to embark on groundbreaking experiments at the Large Hadron Collider (LHC). Among the multitude of processes studied, a specific focus lies on the production of light particles and photons. This endeavour delves into the realm of particle interactions, shedding light on the intricate mechanisms that govern our universe. In this study, we delve into the details of this particular process, exploring data generation, transformation, and classification.

The core of our investigation involves simulating events that encapsulate the production of particles and photons. Two pivotal types of events are generated, each holding distinct significance: Signal and Background Events. Signal events are basically the events of interest to us. For examples, if some new particles are formed from a collision that are new and unknown to us, that would signify some new physics phenomenon that is involved so that would be a signal event. A background event is something that mimics a signal event that could come from a known physical process which we are not interested in. Our main focus is on building classification algorithms to classify a signal event image from a background event image accurately to help discover new phenomena. Till now we have worked on labelled data and build 2 algorithms for the classification- pca and cnn. And currently we are working on cnn.

### PCA Model
#### Loading Images:
1. A function get_images is defined to collect image objects, their names, and labels from a specified folder. It reads images using OpenCV, flattens them, and stores them along with their filenames and labels.
2. Images and labels for signal and background categories are collected using the get_images function for two different folders.
3. The collected images and labels are combined and converted to NumPy arrays for further processing.

#### Eigenface Model:
1. The code then defines a function Eigenface_model for performing Eigenface analysis. It computes the mean face, subtracts it from all faces, calculates the covariance matrix, and obtains eigenvalues and eigenvectors.
2. The eigenvectors are normalized, and the function returns eigenvalues, normalized eigenvectors, the mean face, and the original set of images.

#### Dimension Reduction:
1. The main program uses the Eigenface_model function to obtain eigenvalues, eigenvectors, mean face, and original images.
2. The code then identifies the number of eigenfaces needed for dimension reduction to 95% of the total variance from which we got 17 eigenfaces.

#### Weight Calculation:
1. The projections of the training images onto eigenfaces are computed (weights). These weights will be used to represent each training image in the reduced-dimensional space defined by the selected eigenfaces.

#### Eigenface Test:
1. The code defines a function Eigenface_test for testing the trained model on a set of test images. Test images are normalized and projected onto the selected eigenfaces.
2. For each test image, the code calculates the sum of squared errors with respect to the training images' weights. The closest matching face is determined based on the minimum sum of squared errors.
3. The code checks if the predicted class matches the true class and prints the result for each test image. It calculates and prints the final accuracy of around 96% of the Eigenfaces model on the test set.











