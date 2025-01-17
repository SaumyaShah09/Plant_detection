import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import os

model_path = r'C:\Users\sayhe\OneDrive\Desktop\7. Project-6 Project On Plant Disease Prediction\6.2 Plant Disease Flask App\Plant Disease Flask App\Plant_Disease\plant_disease.h5'

if os.path.exists(model_path):
    print("Model file found!")
else:
    print("Model file not found!")

# Initialize model variable
model = None

# Check if the model file exists
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Model file does not exist. Please check the path.")

# Name of the classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Streamlit app title
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Image uploader widget
plant_image = st.file_uploader("Choose an image...", type="jpg")

# Prediction button
submit = st.button('Predict')

# If the predict button is clicked
if submit:
    if model is not None:
        if plant_image is not None:
            # Convert the uploaded image to a NumPy array
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Display the uploaded image
            st.image(opencv_image, channels="BGR")

            # Check the shape of the image
            print(f"Image shape: {opencv_image.shape}")

            # Resize the image to fit the model input size
            opencv_image = cv2.resize(opencv_image, (256, 256))

            # Normalize the image (if needed)
            opencv_image = opencv_image / 255.0

            # Reshape the image to match the model input format (batch size, height, width, channels)
            opencv_image = np.expand_dims(opencv_image, axis=0)

            # Make prediction
            Y_pred = model.predict(opencv_image)
            predicted_class = CLASS_NAMES[np.argmax(Y_pred)]

            # Display the prediction result
            st.title(f"This is a {predicted_class.split('-')[0]} leaf with {predicted_class.split('-')[1]}")
        else:
            st.error("Please upload an image to proceed.")
    else:
        st.error("Model not loaded. Please check the model path and ensure the model file exists in it.")
