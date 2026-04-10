import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from PIL import Image
import io

IMG_SIZE = 224
classes = ["Nondemented", "Verymilddemented",
           "MildDemented", "Moderatedemented"]

@st.cache_resource
def load_models():
    svm = joblib.load("svm_local.pkl")
    pca = joblib.load("pca_local_zlib.pkl")
    # VGG16 expects (224, 224, 3)
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    vgg = Model(inputs=base_model.input, outputs=base_model.output)
    return svm, pca, vgg

def is_valid_mri(img_array):
    """
    Checks if the image is likely an MRI (grayscale-ish) vs a normal colored photo.
    """
    # 1. Convert to float for calculation
    img_float = img_array.astype(np.float32)
    
    # 2. Split channels
    b, g, r = cv2.split(img_float)
    
    # 3. Calculate mean absolute difference between channels
    # In a true grayscale/MRI image, R, G, and B are almost identical.
    diff_rg = np.mean(np.abs(r - g))
    diff_gb = np.mean(np.abs(g - b))
    
    # Threshold: If the average difference between channels is > 8-10, it's likely a color photo.
    if diff_rg > 10 or diff_gb > 10:
        return False
    return True

st.set_page_config(page_title="Alzheimer's Detection", page_icon="🧠")
st.title("🧠 Alzheimer's Detection App")
st.write("Upload a **Brain MRI** image (Grayscale) for dementia stage prediction.")

uploaded_file = st.file_uploader(
    "Choose MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- STEP 1: LOAD AND NORMALIZE CHANNELS ---
    # .convert('RGB') ensures that even grayscale JPEGs are read as 3-channel (R=G=B)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to cv2 (BGR)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # --- STEP 2: MRI VALIDATION GATE ---
    if not is_valid_mri(img_array):
        st.error("❌ **Invalid Image Detected.** This model is designed for Brain MRI scans only. Please upload a grayscale MRI image.")
    else:
        # --- STEP 3: PRE-PROCESSING ---
        img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img_input = np.expand_dims(img, axis=0)

        svm_model, pca_model, vgg_model = load_models()

        with st.spinner("Extracting features and predicting..."):
            # Feature extraction using VGG16
            features = vgg_model.predict(img_input)
            features = features.reshape(1, -1)
            
            # Dimensionality reduction
            features_pca = pca_model.transform(features)
            
            # Classification
            prediction = svm_model.predict(features_pca)
            probability = svm_model.predict_proba(features_pca)[0]

        # --- STEP 4: RESULTS ---
        st.subheader("📊 Prediction Results")
        predicted_class = classes[prediction[0]]
        st.success(f"**Predicted Stage: {predicted_class}**")

        st.subheader("📈 Confidence Scores")
        # Create columns for better layout
        cols = st.columns(len(classes))
        for i, (label, prob) in enumerate(zip(classes, probability)):
            cols[i].metric(label, f"{prob*100:.1f}%")

        # Visualizing with Matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'] # Green to Red
        bars = ax.bar(classes, probability*100, color=colors)
        
        ax.set_ylabel("Probability (%)")
        ax.set_title("Model Confidence Distribution")
        ax.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)

st.divider()
st.info("💡 Note: This tool uses a VGG16+SVM pipeline. High confidence in 'Nondemented' for a non-brain image is avoided by the color-check filter.")
