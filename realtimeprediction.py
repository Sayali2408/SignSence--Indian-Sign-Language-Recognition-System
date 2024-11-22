import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import logging
from datetime import datetime

# Set up logging for model predictions
logging.basicConfig(filename="gesture_predictions.log", level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Load the trained model
model = load_model("trainedModel.keras")

# Define class names manually (this should correspond to the output layer of your model)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
               'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Ensure the order matches your model's output

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Define the size of the region of interest (ROI) for gesture capture
roi_start_x, roi_start_y, roi_end_x, roi_end_y = 100, 100, 300, 300

# Frame rate control (FPS)
fps = 30  # You can change this to control how fast predictions are made
frame_time = 1.0 / fps
prev_time = time.time()

# Function to preprocess the image
def preprocess_image(frame, roi_start_x, roi_start_y, roi_end_x, roi_end_y):
    """Preprocesses the input image for model prediction."""
    cropped_frame = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    img = cv2.resize(cropped_frame, (64, 64))  # Resize the cropped frame
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict gesture
def predict_gesture(model, img):
    """Makes prediction using the trained model."""
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    class_label = class_names[class_index]
    confidence = np.max(prediction)
    return class_label, class_index, confidence

# Initialize the frame counter
frame_count = 0

# Start capturing video feed and make predictions
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Record frame count and time to manage FPS
    frame_count += 1
    current_time = time.time()
    
    # Ensure that the prediction happens only once per frame rate interval
    if current_time - prev_time >= frame_time:
        prev_time = current_time
        
        # Preprocess the current frame
        img = preprocess_image(frame, roi_start_x, roi_start_y, roi_end_x, roi_end_y)

        # Make prediction
        class_label, class_index, confidence = predict_gesture(model, img)
        
        # Log the prediction
        logging.info(f"Predicted: {class_label} (Class Index: {class_index}), Confidence: {confidence:.2f}")
        
        # Display prediction on the frame (updated to black color)
        cv2.putText(frame, f"Label: {class_label} (Class: {class_index}) ({confidence:.2f})", 
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw rectangle to guide hand placement
    cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (0, 255, 0), 2)

    # Display the frame with prediction
    cv2.imshow('ISL Recognition', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Final log entry after exiting the program
logging.info(f"Exited program at {datetime.now()}")
