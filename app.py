import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io

# Load pre-trained model
@st.cache_resource
def load_model_from_file():
    model = load_model("Garbage-Classification-App\model.h5")  # Update the path if necessary
    return model

model = load_model_from_file()

# Define class mapping
class_labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "About"])

if page == "Home":
    # Main App Title
    st.title("Garbage Classification App")
    st.write("Upload an image, and the app will classify it into one of six garbage categories.")

    # Upload Image Section
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the Image
        def preprocess_image(img):
            img = img.resize((32, 32))  # Resize to 32x32 pixels
            img_array = img_to_array(img, dtype=np.uint8)  # Convert to array
            img_array = np.array(img_array) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array

        processed_image = preprocess_image(image)

        # Prediction with Spinner
        with st.spinner("Classifying..."):
            predictions = model.predict(processed_image)
        confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions, axis=-1)[0]
        predicted_class = class_labels[predicted_class_index]

        # Display Results
        st.success(f"### Predicted Class: **{predicted_class}**")
        st.write(f"### Confidence: **{confidence:.2f}**")

        # Display Class Probabilities in a Table
        st.write("### Class Probabilities")
        probabilities = pd.DataFrame({
            "Class": list(class_labels.values()),
            "Probability": predictions[0]
        })
        st.table(probabilities)

        # Display a Bar Chart for Probabilities
        st.write("### Class Probabilities Chart")
        st.bar_chart(probabilities.set_index("Class"))

        # Allow Image Download
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="classified_image.jpg",
            mime="image/jpeg"
        )

        # Plot Processed Image
        st.write("### Processed Image")
        fig, ax = plt.subplots()
        ax.imshow(processed_image[0])
        ax.axis("off")
        ax.set_title(f"Prediction: {predicted_class}")
        st.pyplot(fig)

elif page == "About":
    st.title("About the Garbage Classification App")
    st.write("""
        This app uses a pre-trained neural network model to classify images into six categories of garbage:
        - Cardboard
        - Glass
        - Metal
        - Paper
        - Plastic
        - Trash

        ### How It Works:
        - Upload an image of garbage.
        - The app preprocesses the image and predicts its class using a neural network.
        - The prediction and class probabilities are displayed, along with visualizations.

        Created with ❤️ using Streamlit and TensorFlow.
    """)
