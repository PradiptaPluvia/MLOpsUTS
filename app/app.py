# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Simple page config
st.set_page_config(page_title="Shoe Classifier", page_icon="👟")

st.title("👟 Shoe vs Sandal vs Boot Classifier")
st.write("Upload a picture and I'll tell you if it's a shoe, sandal, or boot!")

class SimpleClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['Boot', 'Sandal', 'Shoe']
        self.img_size = 64
    
    def load_model(self):
        try:
            self.model = tf.keras.models.load_model('simple_shoe_model.h5')
            return True
        except:
            return False
    
    def predict(self, image):
        # Convert and resize image
        img = np.array(image)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return self.class_names[predicted_class], confidence

# Load model
classifier = SimpleClassifier()

if not os.path.exists('simple_shoe_model.h5'):
    st.error("Model file 'simple_shoe_model.h5' not found! Train the model first.")
    st.stop()

if not classifier.load_model():
    st.error("Failed to load model!")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your image", use_column_width=True)
    
    # Predict
    with st.spinner("Thinking..."):
        predicted_class, confidence = classifier.predict(image)
    
    # Show result
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.1%}")
    
    # Simple confidence bar
    st.progress(float(confidence))