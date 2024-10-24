Fish Species Classification with Deep Learning
Project Overview
This project aims to classify 9 different species of fish using deep learning techniques. The dataset, sourced from the A Large Scale Fish Dataset, contains images of fish that are categorized into predefined species. The model is designed to accurately predict the species of fish based on the images.
Project Structure
Data Preprocessing
Loaded image file paths and labels.
Data augmentation techniques such as rotation, shifting, and flipping were applied to improve the model’s ability to generalize.
Model Development
Built an Artificial Neural Network (ANN) using Keras and TensorFlow.
The model consists of multiple dense layers, batch normalization, and dropout layers to prevent overfitting.
Loss function: Categorical Crossentropy
Optimizer: Adam
Training & Validation
The dataset was split into training, validation, and test sets.
EarlyStopping and ReduceLROnPlateau were used to optimize the training process and prevent overfitting.
The model was trained for 20 epochs.
Evaluation
Evaluated the model’s performance using accuracy, a confusion matrix, and classification reports.
Kaggle Notebook
You can view the project and run it on Kaggle using the link below:
https://www.kaggle.com/code/aslemimolu/classification-in-a-large-scale-fish-data-set






Contributors




Aslı Şemşimoğlu




Rabia Durgut



