### AGROASSIST - A Agriculture AI application

AgroAssist is an AI-powered application designed to support and optimize agricultural practices. By leveraging advanced machine learning and deep learning techniques, AgroAssist aims to assist farmers with accurate crop recommendations and timely disease detection for improved productivity and sustainable farming.

# Features 

1. Crop Recomendation 
2. Diease Detection 
3. Pest Identifiction


### Crop Recommendation Using Machine Learning

Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. This application will assist farmers in increasing agricultural productivity, preventing soil degradation in cultivated land, reducing chemical use in crop production, and maximizing water resource efficiency.

# [Dataset]()
This dataset was build by augmenting datasets of rainfall, climate and fertilizer data available for India.

### [Attributes information:]()
* **N** - Ratio of Nitrogen content in soil
* **P** - Ratio of Phosphorous content in soil
* **K** - Ratio of Potassium content in soil
* **Temperature** -  temperature in degree Celsius
* **Humidity** - relative humidity in %
* **ph** - ph value of the soil
* **Rainfall** - rainfall in mm 

# [Experiment Results:]()
* **Data Analysis**
    * All columns contain outliers except for N.
 * **Performance Evaluation**
    * Splitting the dataset by 80 % for training set and 20 % validation set.
 * **Training and Validation**
    * GausianNB gets a higher accuracy score than other classification models.
    * GaussianNB ( 99 % accuracy score )
 * **Performance Results**
    * Training Score: 99.5%
    * Validation Score: 99.3%

 
### Recognition of Plant Diseases 


The traditional method of disease detection has been to use manual examination by either farmers or experts, which
can be time consuming and costly, proving infeasible for millions of small and medium sized farms around the world. This feature is an approach to the development of plant disease recognition model, based on leaf image classification, by the use of deep convolutional networks. The developed model is able to recognize 38 different types of plant diseases out of of 14 different plants with the ability to distinguish plant leaves from their surroundings.

This process for building a model which can detect the disease assocaited with the leaf image. The key points to be followed are:

1. Data gathering

   The dataset taken was **"New Plant Diseases Dataset"**. It can be downloaded through the link "https://www.kaggle.com/vipoooool/new-plant-diseases-dataset". It is an Image dataset containing images of different healthy and unhealthy crop leaves.

2. Model building

   - I have used pytorch for building the model.
   - I used three models:-
     1. The CNN model architecture consists of CNN Layer, Max Pooling, Flatten a Linear Layers.
     2. Using Transfer learning VGG16 Architecture.
     3. Using Transfer learning resnet34 Architecture.

3. Training

   The model was trained by using variants of above layers mentioned in model building and by varying hyperparameters. The best model was able to achieve 98.42% of test accuracy.

4. Testing

   The model was tested on total 17572 images of 38 classes.


### Pest Classification

AgroAssist’s Pest Classification feature provides a deep learning-based approach to identifying common agricultural pests, helping farmers implement effective pest control measures. By accurately classifying pest species from images, this feature enables timely intervention, ultimately reducing crop damage and minimizing pesticide use.

# Data Preparation:

Images are preprocessed and augmented using TensorFlow's ImageDataGenerator with an 80/20 split for training and validation.

# Model Architecture:

Base Model: Utilizes ResNet50, a pre-trained deep convolutional network, as the foundation for feature extraction.
Custom Layers: Adds custom layers for improved classification accuracy, including:
Global Average Pooling Layer
Dense layer with 1024 units and ReLU activation
Output layer with softmax activation to classify pest species.

# Training:

The model is initially trained with the base ResNet50 layers frozen, using a learning rate of 0.001.
Fine-Tuning: After initial training, the last 20 layers of ResNet50 are unfrozen, and the model is retrained with a reduced learning rate of 0.00001 to further enhance accuracy.

# Results:

The trained model achieved high test accuracy of 69% , demonstrating reliable pest identification.

