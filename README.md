# **Transfer Learning and Classification Projects**

This repository contains multiple Jupyter Notebooks focused on **Transfer Learning**, classification tasks, and feature extraction using various datasets and deep learning techniques. These notebooks demonstrate the implementation of state-of-the-art models such as **ResNet**, **VGG**, and other architectures fine-tuned to specific datasets.

## **Contents**

Here is a brief description of each notebook:

1. **04_Transfer_Learning.ipynb**  
   - Basic implementation of transfer learning using pre-trained models to classify images in food vision dataset.
   
2. **ResnetV250_(simpson_dataset).ipynb**  
   - Fine-tuning a pre-trained **ResNetV2** model for image classification on the **Simpson Characters Dataset**.
   
3. **Transfer_learning_fine_tuning.ipynb**  
   - A deeper dive into transfer learning by fine-tuning the pre-trained models for better performance on food vision dataset.
   
4. **bird_species.ipynb**  
   - Using transfer learning to classify various species of birds. The model is trained and fine-tuned on a bird species dataset.
   
5. **brain_tumour_classification.ipynb**  
   - Classification of brain tumor MRI images using transfer learning with custom CNN models.
   
6. **cancer.ipynb**  
   - A binary classification task using machine learning models to detect cancer from medical image datasets.
   
7. **feature_extraction.ipynb**  
   - Demonstrates feature extraction using pre-trained deep learning models. Useful for downstream tasks such as clustering or classification.
   
8. **insurance.ipynb**  
   - Machine learning model predicting insurance cost using demographic data. Although unrelated to transfer learning, this notebook provides insight into feature engineering and classical ML.
   
9. **transfer_learning_scaling.ipynb**  
   - Experimentation with scaling the number of training samples while using transfer learning, illustrating the robustness of pre-trained models with smaller datasets.

## **Key Concepts**

- **Transfer Learning**: 
   - Reusing a pre-trained model on a new problem by transferring the learned features to a new task.
   - Reduces training time and improves model performance, especially with limited labeled data.
   
- **Fine-Tuning**: 
   - Further training a pre-trained model by unfreezing some of its layers and adapting it to a new dataset.
   
- **Feature Extraction**: 
   - Extracting high-level features from images using pre-trained convolutional layers, which can then be used for custom classification tasks.

## **Getting Started**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ifrahaha/Deep-Neural-Network-using-Tensorflow.git
   cd Deep-Neural-Network-using-Tensorflow
   
2. **Set up the environment**:

You will need Python and some essential libraries like TensorFlow, Keras, OpenCV, and Matplotlib.
Install the dependencies using:
```bash
pip install tensorflow keras opencv-python matplotlib
