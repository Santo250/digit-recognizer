import streamlit as st
import cv2
import numpy as np
import base64
import json

st.title("Digit Recognizer")

# Load the ONNX model
@st.cache_resource
def load_model():
    return cv2.dnn.readNetFromONNX('model.onnx')

net = load_model()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def predict_digit(image):
    # Resize and normalize
    img = cv2.resize(image, (28, 28))
    blob = cv2.dnn.blobFromImage(img, 1/255, (28, 28))
    net.setInput(blob)
    out = net.forward()
    out = softmax(out.flatten())
    class_id = np.argmax(out)
    confidence = out[class_id]
    return class_id, confidence

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Make prediction
    class_id, confidence = predict_digit(img)
    
    st.success(f"Predicted Digit: {class_id}")
    st.info(f"Confidence: {confidence:.2%}")