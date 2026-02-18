import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.title("✏️ Digit Recognizer")
st.write("Draw a digit (0-9) in the canvas below and the model will predict it!")

# Load the ONNX model
@st.cache_resource
def load_model():
    return cv2.dnn.readNetFromONNX('model.onnx')

net = load_model()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def predict_digit(image):
    # Resize to 28x28
    img = cv2.resize(image, (28, 28))
    # Create blob
    blob = cv2.dnn.blobFromImage(img, 1/255, (28, 28))
    net.setInput(blob)
    out = net.forward()
    out = softmax(out.flatten())
    class_id = np.argmax(out)
    confidence = out[class_id]
    return class_id, confidence

# Create drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("🔮 Predict Digit"):
    if canvas_result.image_data is not None:
        # Get the image from canvas
        img = canvas_result.image_data.astype(np.uint8)
        
        # Convert RGBA to grayscale
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Invert colors (model expects white digit on black background)
        gray = 255 - gray
        
        # Make prediction
        class_id, confidence = predict_digit(gray)
        
        # Display results
        st.success(f"### Predicted Digit: {class_id}")
        st.info(f"### Confidence: {confidence:.2%}")
        
        # Show processed image
        st.image(gray, caption="Processed Image (28x28)", width=200)
    else:
        st.warning("Please draw a digit first!")

# Clear canvas button
if st.button("🗑️ Clear Canvas"):
    st.rerun()