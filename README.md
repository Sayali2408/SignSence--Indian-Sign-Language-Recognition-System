# SignSense--Indian-Sign-Language-Recognition-System

OBJECTIVE
The objective of this project is to develop a robust system for the analysis and recognition of various alphabets and numbers from a comprehensive dataset of Indian Sign Language (ISL) images. The dataset encompasses a diverse collection of images captured under varying lighting conditions and featuring a wide range of hand orientations and shapes, ensuring adaptability and reliability in real-world scenarios.

PRE-REQUISITES
Before running this project, make sure you have following dependencies -

Python

Pip

OpenCV

TensorFlow

Keras

NumPy

STEPS OF EXECUTION
1. Image Collection: The initial step involved capturing hand gesture images representing various letters and numbers in Indian Sign Language (ISL). A simple camera interface was developed using OpenCV to acquire thousands of images for each sign. The dataset was designed to include multiple samples per sign to ensure the model’s robustness and generalizability.  

2. Data Splitting: Following data collection, the images were organized into training and validation sets. This split was crucial to facilitate effective learning while enabling accurate evaluation on unseen data, reducing the risk of overfitting and ensuring reliable model performance assessment.  

3. Data Preprocessing: To prepare the images for model training, several preprocessing steps were applied. Images were resized to a uniform dimension, pixel values were normalized, and, where applicable, images were converted to grayscale. Additionally, data augmentation techniques, such as rotation and flipping, were employed to enhance dataset diversity and improve the model’s generalization capabilities.  

4. Model Construction: A Convolutional Neural Network (CNN) was designed to classify images into their respective sign classes. The architecture included multiple convolutional layers to extract spatial features from hand gestures, pooling layers to reduce dimensionality, and dense layers to perform the final classification.  

5. Model Training: The CNN was trained on the prepared dataset, with its progress monitored using the validation set. Over successive epochs, the model refined its understanding of the distinguishing features of each sign. Hyperparameter tuning was conducted to optimize performance, ensuring high accuracy in classification tasks.  

6. Real-Time Prediction: After training, the model was integrated with a real-time camera feed for live gesture recognition. Each frame was captured, preprocessed, and fed into the trained model for prediction. The recognized sign was displayed instantly, providing immediate feedback to users.

Group Members
