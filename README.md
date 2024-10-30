# Fish Species Classification with Deep Learning

## Project Overview

This project aims to classify 9 different species of fish using deep learning techniques. The dataset, sourced from the A Large Scale Fish Dataset, contains images of fish that are categorized into predefined species. The model is designed to accurately predict the species of fish based on the images.

## Project Structure

### 1. Data Preprocessing

* Loaded image file paths and labels.
* Data augmentation techniques such as rotation, shifting, and flipping were applied to improve the model’s ability to generalize.

### 2. Model Development

* Built an Artificial Neural Network (ANN) using Keras and TensorFlow.
* The model consists of multiple dense layers, batch normalization, and dropout layers to prevent overfitting.
* *Loss function*: Categorical Crossentropy
* *Optimizer*: Adam

### 3. Training & Validation

* The dataset was split into training, validation, and test sets.
* EarlyStopping and ReduceLROnPlateau were used to optimize the training process and prevent overfitting.
* The model was trained for 20 epochs.

### 4. Evaluation

* Evaluated the model’s performance using accuracy, a confusion matrix, and classification reports.

### 5. Hyperparameter Optimization Strategy
In training the model, several hyperparameters were carefully chosen to optimize learning performance and prevent overfitting. These include settings for learning rate, dropout rate, batch size, and the patience levels for callback functions. Here’s a detailed explanation of each:

 * Learning Rate (Adam Optimizer):

Learning Rate: 0.0005
The initial learning rate of 0.0005 was selected to provide a balance between speed and stability in learning. Using a smaller learning rate, we ensure that the model makes more refined adjustments to the weights in each epoch, reducing the risk of overshooting the minimum loss.
 * Batch Size:

Batch Size: 256
A batch size of 256 was chosen to allow the model to learn from a sufficiently large sample of images per step, which helps in generalizing the model. Larger batch sizes can improve memory efficiency, while a well-chosen batch size like 256 enables the model to learn from diverse samples within each batch, reducing overfitting.
 * Dropout Rates:

Dropout Rate: 0.2 after each dense layer
Dropout with a rate of 0.2 is used after each dense layer to randomly disable 20% of the neurons in the network during each training step. This regularization technique helps to prevent overfitting by ensuring that the model does not rely too heavily on any particular set of neurons, allowing for better generalization on unseen data.
 * EarlyStopping Callback:

Patience: 5
The EarlyStopping callback monitors the validation loss and stops training if it does not improve for 5 consecutive epochs. This strategy prevents the model from overfitting by halting training once the validation loss plateaus, ensuring optimal model performance without unnecessary epochs.
 * ReduceLROnPlateau Callback:

Factor: 0.2, Patience: 7, Minimum Learning Rate: 1e-6
To adapt the learning rate based on the model's progress, the ReduceLROnPlateau callback reduces the learning rate by a factor of 0.2 if there’s no improvement in validation loss for 7 epochs. This gradual reduction allows the model to fine-tune the weights when nearing convergence, ensuring it doesn’t miss local minima that a fixed learning rate might skip over. The minimum learning rate of 1e-6 acts as a threshold to avoid excessively low learning rates, which could hinder further learning.

### 5. Kaggle Notebook

You can view the project and run it on Kaggle using the link below:
[Classification in a Large Scale Fish Dataset]

([https://www.kaggle.com/code/aslemimolu/classification-in-a-large-scale-fish-data-set])


## Contributors

* Aslı Şemşimoğlu
* Rabia Durgut
