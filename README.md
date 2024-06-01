# Data-Analysis-of-particle-collisions-at-Large Hadron Collider

## Project: Exploring Beyond the Standard Model with High-Energy Physics

### Introduction and Motivation
- High-energy physics endeavours to understand fundamental nature through particle collisions at extreme energies.
- LHC experiments (ATLAS, CMS) amassed vast data; the Standard Model explains only 4% of the universe.
- Pursuit of physics models beyond the Standard Model seeks to unravel the unknown phenomena.

### Background
The pursuit of unravelling the mysteries of the universe has led physicists to embark on groundbreaking experiments at the Large Hadron Collider (LHC). Among the multitude of processes studied, a specific focus lies on the production of light particles and photons. This endeavour delves into the realm of particle interactions, shedding light on the intricate mechanisms that govern our universe. **In this study, we delve into the details of this particular process, exploring data generation, transformation, and classification**.

The core of our investigation involves simulating events that encapsulate the production of particles and photons. Two pivotal types of events are generated, each holding distinct significance: Signal and Background Events. **Signal events** are basically the events of interest to us. For examples, if some new particles are formed from a collision that are new and unknown to us, that would signify some new physics phenomenon that is involved so that would be a signal event. A **background event** is something that mimics a signal event that could come from a known physical process which we are not interested in. **Our main focus is on building classification algorithms to classify a signal event image from a background event image accurately to help discover new phenomena**. Till now we have worked on labelled data and build 2 algorithms for the classification- Principal Component Analysis (PCA) and Convolutional Neural Newtwork (CNN). And currently we are working on CNN.

### Project Goal
- Investigate precision in the SM via the LHC's potential.
- Discover novel physics and maintain background rejection.
- Effective preprocessing(blurring and Deblurring) of the input images using autograd for accurate classification of signal and background events.
- Employ Machine Learning (ML) and Deep Learning algorithms to enhance signal efficiency.


## Classification (Simulated Images):
### Principal Component Analysis (PCA) Model
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
3. The code checks if the predicted class matches the true class and prints the result for each test image. It calculates and prints the final **accuracy of around 96%** of the Eigenfaces model on the test set.

### Convolutional Neural Network (CNN) Model
Transitioning to CNNs, we used deep learning in image analysis. CNNs offer a sophisticated approach to image processing, leveraging hierarchical feature extraction to discern patterns and relationships within images. This architecture's convolutional layers efficiently captured local correlations by convolving filters across the input image, effectively reducing the complexity of handling large image vectors.It first uses the method of convolution wherein a filter of smaller dimensions, say $D \times D$ (where $D < N$) is put on the input image and traversed throughout the image resulting in dot product values as the new image. In our model, we have used the zero padding method where the output image will be of dimensions $(N+D-1, N+D-1)$. After this step, pooling is also done which is another way of reducing dimensions and retaining important information. We have used the max-pooling method here which picks the maximum value in a pooling filter area. In the end, the vectors are put in a dense, fully connected network for a final output of the class. The model we implemented had 20 layers.

#### Dataset:
We used balanced dataset of 200 simulated images of signal and background events. We trained on 80% of images using validation set and tested on the rest.

<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/28cbd00a-3145-409e-a899-c03d766d38e2" width=30% height=30%>

#### CNN Architecture Used:
![cnnarch](https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/37b18d97-929c-41e3-805b-3bc947c851f5)
### Result:

## Pre-processing as an optimization problem:
- Effective preprocessing (blurring and deblurring) of the input images for accurate classification of signal and background events : For this task, we compare results of two python libraries- Numpy and Autograd.
- Our pre-processing task includes devising a method to reduce the noisiness (indicated by grainy appearance of an image) while also preserving the important and fine details and information contained in the image.

To pre-process our jet images, we are using a generic methodology for reconstructing images from recorded data and formulating the image reconstruction problem as the following optimisation problem:
$$C(g_{in}) = C_1 + C_2 = \| g_{out} - \hat{A}g_{in}\|^2 + \alpha \Psi(g_{in})$$
where the terms signify the following:\
C1: Mean square term\ C2: Constraint term\
$\hat{A}$: Operator which models Physics of field propagation from object space to detector data space.\
$\alpha$: Positive constant (Lagrange multiplier) which decides the relative weight between the two terms. This is a hyperparameter.\
$\Psi(g_{in})$: Penalty Function\
We design a penalty such that it has higher value if the graininess in an image is higher. This penalty is known as Total Variation (or TV):
$$\Psi(g) = \int\int dx dy |\nabla g|$$
We used **Gradient Descent Algorithm** to minimise our objective function. Given an image $g$, we want to vary it to $g + \delta g$ (iteratively) such that $C(g)>C(g + \delta g)$ till we find a minimum. $\delta g$ is an update on the image at each pixel. Therefore, functional gradient $\Delta_{g_{in}} C(g_{in})$ is needed in order to find update $\delta g$ which will reduce $C(g_{in})$. The functional gradient is defined by the following relation:
$$\nabla_{g_{in}} C(g_{in}) = - 2 \hat{A}^T (g_{out} - \hat{A}g_{in}) - \alpha \nabla [\frac{\nabla g_{in}}{\sqrt{|\nabla g_{in}|^2 + \epsilon^2}}]$$\
**Defining $g_{in}$ and $g_{out}$:** We took our original jet images and applied Weiner Filter on it after blurring. $g_{in}$ is our weiner filtered image. Weiner filter is a filter used in signal processing to produce an estimate of a desired image. But it has a lot of noise. We want to reduce this noise by taking our $g_{out}$ as the blurred image. The penalty function does not allow $g_{in}$ (noisy image) to completely go towards the blurred image and keep some graininess. So we want a trade-off whereby giving up a bit on error, we want the noisy
appearance to go away. At the same time, we do not just want to blur the features.\
To compute the functional gradient for gradient descent, $\Delta_{g_{in}} C(g_{in}) $, we employed two of the Python libraries- Numpy and Autograd. 

**Numpy Implementation:** We use the symbolic expressions of the derivative of $C(g_{in})$ and compute the gradient using numpy.gradient() function by utilizing either the first or second-order correct one-sides (in either direction) differences at the boundaries and second-order accurate central differences in the interior locations.

**Autograd Implementation:** The conventional method for computing the derivative $\delta C / \delta x_{i}$ involves adjusting $x_{i}$ by a small increment $\delta x_{i}$ and calculating the corresponding change $\delta C$. However, this approach requires running the computational process individually for each input $x_{i}$, which can be computationally expensive. There is a more efficient way to compute derivatives by using Automatic Differentiation technique using autograd package in python. Automatic differentiation (AD) is a method to get exact derivatives efficiently, by storing information as you go forward that you can reuse as you go backwards. It is able to compute an approximation of the derivative of a function, without computing a symbolic expression of the derivative and with machine precision accuracy. In essence, the algorithm simplifies a complex function into a series of simpler functions. It then computes the derivatives of these simpler functions, starting from the innermost one and working outwards. This process, known as "reverse expansion", ensures that the gradient of each function is calculated and combined until the original function's gradient is determined. Autograd is an Automatic Differentiation (AD) software library which uses this method to compute gradients. As mentioned before, it uses reverse-mode differentiation (a.k.a. backpropagation) to efficiently compute gradients of functions written in plain Python/Numpy. Machine learning research often boils down to creating a loss function (Objective function) and optimizing it with gradients. Autograd lets us write down the loss function using the full expressiveness of Python/Numpy and get the gradient with one function call.
### Results:
The original image used for application of the pre-processing is given below.\
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/58d7aba9-9b0e-4bc1-8bb0-0b9538198512" width=30% height=30%>

Further, we had $g_{out}$: Blurred image and $g_{in}$: Weiner image as shown below:\
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/3f38b392-5dbc-4fe2-b160-fecf47e9d6cc" width=30% height=30%>
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/69ed21a3-3108-47c6-b6b2-e95b5cd15c54" width=30% height=30%>

We implemented the gradient descent for TV optimization via two methods, using the standard Numpy library and using Autograd. The final images of both is given below for comparison:\
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/3fe67351-4b12-406f-9b43-0812a57811dd" width=30% height=30%>
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/96aa89a7-3e21-42cf-93f7-712865dd9bd8" width=30% height=30%>

On comparison of the two images, it was found that the **Autograd method was more efficient** and its final image had better signal-to-noise ratio. After determining this, we implemented the algorithm on our Jet images.\
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/28cbd00a-3145-409e-a899-c03d766d38e2" width=30% height=30%>
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/97cca925-ef57-42a1-94d9-ccb11ecf6624" width=30% height=30%>
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/5bae1817-cef5-412c-8f58-7bdc4bfd6145" width=30% height=30%>
<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/554aabff-d082-41c0-b0b9-0719b364d1df" width=30% height=30%>

### Distance Metrics:
- Calculating the sum of squared distance of the final Autograd image to the other images gave the following result:

                  SSQ distance Autograd from blurred image:  7.40 e-05
                  SSQ distance Autograd from Original image:  6.25 e-06
                  SSQ distance Autograd from Weiner image:  1.07 e-08
- To quantify how much the Autograd pre-processing improves the classification of the images into signal and background, we use the same **SSQ distance metric**. We calculated the distance between the mean of 200 Signal images and mean of 200 Background images in Weiner and Autograd:

                  Distance between the two classes in Weiner images: 0.0915
                  Distance between the two classes in Autograd images: 0.0913

- For a more accurate determination of the power of autograd, we used another distance metric for the purpose. It is called the **Bhattacharyya distance**. This distance measure has numerous applications in fields ranging from machine learning and statistics to data science and image processing. The Bhattacharyya distance is a way of quantifying the differences between two probability distributions. It tells us how much overlap there is between the two distributions, and can help us determine how similar or dissimilar they are. The distance is based on the Bhattacharyya coefficient, which measures the degree of overlap between two statistical samples or populations and requires the discrete histograms of the two distributions.\ The Bhattacharyya distance between two discrete probability distributions $P$ and $Q$ is calculated by the following formula:
    $$D_B(P,Q) = -ln(BC(P,Q))$$
where $BC(P,Q)$ is the Bhattacharyya coefficient, which is defined as:
    $$BC(P,Q) = \sum_{i=1}^{N} \sqrt{P(x_i)Q(x_i)}$$
In our case, the two probability distributions $P$ and $Q$ are histograms of mean images of 200 signal and background images each in both Weiner and Autograd implementation. To find the Bhattacharyya distance, we used cv2 module of the Open Source Computer Vision Library in python:

                 Bhattacharyya distance between the two classes in Weiner images: 0.4408
                 Bhattacharyya distance between the two classes in Autograd images: 0.4425
  We can see that the distance is greater in the case of Autograd than in that of Weiner.

## Electron-Photon classification
### Motivation: To find a correlation between an individual particle and its corresponding shower.
The interactions during particle collisions lead to the creation of new particles. These freshly generated particles can undergo subsequent decay processes, splitting into multiple other particles, and also emit energy as they traverse. Consequently, what we observe is not just a solitary particle but rather a cascading shower of particles, each revealing a piece of the intricate dynamics occurring at the subatomic level Ideally, we would like to accurately pinpoint the location of the solitary particle i.e photon or electron from their respective showered images that would help in accurate classification of electron and photon. Hence, our objective was to employ machine learning methods to establish a correlation between an individual particle (electron or photon) and the resulting particle shower, by learning an image filter. This filter, when applied to an image of a single particle, would generate a prediction of its corresponding particle shower. By doing so, we aim to sidestep the intricate quantum mechanics required for directly predicting a particle shower.Initially, we explored various blurring filters that could convert a single spot image into a showered image by blurring. But our goal was to use machine learning techniques and build such models that could learn filters specifically tailored for electrons and photons as these particles theoretically shower differently and thus, they would each generate a unique shower pattern. However, this task proved to be very challenging. As a result, instead of converting showered images into single spot images, we used the showered images themselves to differentiate between electrons and photons. This approach allows us to identify the type of particle based on the characteristics of its showered image, providing valuable information for our analysis.

### Dataset:
We utilized the convolutional neural networks for the electron photon classification. We used the Electron-Photon dataset available on the CERN website for the modeling. In this dataset, electrons are signal images and photons are background images. The dataset had 16,000 images each of electron and photon where every image has dimensions 32 x 32, with two channels, representing Time and Energy. An example is shown below.

![Energy-Time](https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/de4e3a3a-9563-4de0-b952-b07e29cc339d)

### CNN Model:

We trained on 80% of images using validation set and tested on the rest. Our deep learning model was 11 layers deep, ending in a sigmoid function giving us the probability of the class the image belongs to. Below is the visual representation of our cnn architecture.

<img src="https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/9592584f-7329-4c97-b1b5-7a4cf91f7a6b">

### Results:
- **AUC-ROC**- AUC-ROC curve for our CNN model with AUC score = 0.697
  
  ![Untitled](https://github.com/kkirtigarg/Data-Analysis-of-particle-collisions-at-LHC/assets/157001390/6bd25e7e-c228-4434-9cb4-817a8e17c9c1)

- **Accuracy**- Ratio of the number of correct predictions to the total number of predictions made:
  
                                Training accuracy: 74%
                                Testing accuracy: 66%
  



