# SignSense: Indian Sign Language Recognition System  

📌 Objective  
The goal of this project is to develop a robust and reliable system for the recognition of alphabets and numbers from Indian Sign Language (ISL). Using a comprehensive dataset, the system is designed to handle diverse image conditions, including variations in lighting, hand orientations, and shapes, ensuring adaptability for real-world applications.  

---  

🛠️ Pre-requisites 
Before running this project, ensure that you have the following dependencies installed:  

- Python (v3.x or higher)  
- Pip  
- OpenCV  
- TensorFlow  
- Keras 
- NumPy  

To install the required packages, use:  
```bash
pip install -r requirements.txt
```  

---  

🚀 Steps of Execution

1️⃣ Image Collection  
The first step involves capturing hand gesture images representing ISL alphabets and numbers. A simple camera interface was created using OpenCV to collect thousands of images for each sign. The dataset was designed to include multiple samples to ensure the model’s robustness and ability to generalize across variations.  

2️⃣ Data Splitting 
The collected images were split into training and validation sets. This separation helps the model learn effectively from the training data while being evaluated on unseen validation data, ensuring reliable performance and minimizing overfitting risks.  

3️⃣ Data Preprocessing 
The images were preprocessed to prepare them for model training. Key preprocessing steps included:  
- Resizing: Uniform resizing of images to a consistent dimension.  
- Normalization: Scaling pixel values for efficient learning.  
- Data Augmentation: Techniques like rotation and flipping were applied to increase dataset diversity and improve the model’s generalization capabilities.  

4️⃣ Model Building  
A Convolutional Neural Network (CNN) was designed to classify the images into their respective sign classes.  
- Convolutional Layers**: Extract spatial features from the hand gestures.  
- **Pooling Layers**: Reduce the dimensionality of feature maps.  
- **Dense Layers**: Perform the final classification task.  

5️⃣ Model Training
The CNN model was trained on the prepared dataset, with validation data used to monitor its progress.  
- Epochs: The model refined its understanding over multiple iterations.  
- Hyperparameter Tuning: Adjustments were made to optimize the model’s accuracy.  

6️⃣ Real-Time Prediction 
After successful training, the model was integrated with a real-time camera feed.  
- Each frame captured by the camera is **preprocessed** and passed to the trained model for prediction.  
- The **recognized sign** is displayed instantly, providing users with immediate feedback.  

---  

📂 Project Structure 

```
SignSense/
│
├── dataset/                # Collected ISL images (train & validation sets)
├── models/                 # Saved model files
├── scripts/                # Scripts for training, testing, and real-time prediction
│   ├── train.py            # Model training script
│   ├── predict.py          # Real-time prediction script
│   └── preprocess.py       # Data preprocessing and augmentation script
├── requirements.txt        # List of required dependencies
└── README.md               # Project documentation (this file)
```  

---  

💻 How to Run the Project

1. Clone the Repository:  
   ```bash
   git clone https://github.com/your-username/SignSense.git
   cd SignSense
   ```  

2. Install Dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Prepare the Dataset:  
   - Collect ISL images using the OpenCV script or use an existing dataset.  
   - Organize the images into `train` and `validation` directories under the `dataset/` folder.  

4. Train the Model:  
   ```bash
   python scripts/train.py
   ```  

5. Run Real-Time Prediction:  
   ```bash
   python scripts/predict.py
   ```  

---  


🤝 Group Members  
@https://github.com/Sayali2408
@https://github.com/anyalisis12
@https://github.com/Sayali2408
