import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✏️ Digit Recognizer")
st.write("Draw a digit (0-9) in the canvas below. For best accuracy, draw large and centered.")

# Load the ONNX model
@st.cache_resource
def load_model():
    return cv2.dnn.readNetFromONNX('model.onnx')

net = load_model()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def preprocess_image(img):
    """
    Preprocess the canvas image to match MNIST training data.
    1. Convert to grayscale
    2. Threshold to binary (remove anti-aliasing noise)
    3. Find bounding box and crop to digit
    4. Resize to 20x20 (leaving padding)
    5. Place on 28x28 canvas (centering)
    6. Normalize to 0-1 range
    """
    # Convert RGBA to grayscale
    if img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Threshold: Make it pure black and white (removes gray edges)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours to locate the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (the digit)
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop to the digit with some padding
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        
        cropped = thresh[y1:y2, x1:x2]
        
        # Resize to 20x20 (MNIST digits are typically 20x20 on 28x28 canvas)
        resized = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 black canvas and place digit in center
        final = np.zeros((28, 28), dtype=np.uint8)
        offset_x = (28 - 20) // 2
        offset_y = (28 - 20) // 2
        final[offset_y:offset_y+20, offset_x:offset_x+20] = resized
        
        # Normalize to 0-1 range (float32)
        final = final.astype(np.float32) / 255.0
        
        return final, thresh, cropped
    else:
        # Fallback if no digit detected
        gray = cv2.resize(gray, (28, 28))
        return gray.astype(np.float32) / 255.0, thresh, gray

def predict_digit(image):
    # Image should already be 28x28, normalized, grayscale
    # Reshape to (1, 1, 28, 28) for model input
    blob = image.reshape(1, 1, 28, 28)
    net.setInput(blob)
    out = net.forward()
    out = softmax(out.flatten())
    class_id = np.argmax(out)
    confidence = out[class_id]
    return class_id, confidence

# --- Canvas Settings ---
# Stroke width 15-25 is usually best for MNIST models
stroke_width = st.sidebar.slider("Stroke Width", 10, 30, 20)

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=stroke_width,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
)

# --- Prediction Logic ---
if st.button("🔮 Predict Digit", type="primary"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype(np.uint8)
        
        # Preprocess
        processed, thresh, cropped = preprocess_image(img)
        
        # Predict
        class_id, confidence = predict_digit(processed)
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"### {class_id}")
        with col2:
            st.info(f"### {confidence:.1%}")
        
        # Show preprocessing steps for debugging
        st.markdown("### 🔍 Preprocessing Steps")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(img, caption="1. Original Canvas")
        with col_b:
            st.image(thresh, caption="2. Thresholded", clamp=True)
        with col_c:
            st.image(cropped, caption="3. Cropped Digit", clamp=True)
            
        st.image(processed, caption="4. Final 28x28 Input", width=150, clamp=True)
    else:
        st.warning("Please draw a digit first!")

# Sidebar instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips for Accuracy")
st.sidebar.markdown("- Draw the digit **large** in the center")
st.sidebar.markdown("- Use **stroke width 15-25**")
st.sidebar.markdown("- Draw **one digit only**")
st.sidebar.markdown("- Avoid cursive or fancy styles")