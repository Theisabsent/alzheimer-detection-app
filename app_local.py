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
    pca = joblib.load("pca_local.pkl")
    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    vgg = Model(inputs=base_model.input, outputs=base_model.output)
    return svm, pca, vgg


st.title("🧠 Alzheimer's Detection App (Local Trained Model)")
st.write("Upload an MRI brain image for dementia stage prediction.")

uploaded_file = st.file_uploader(
    "Choose MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to cv2
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    svm_model, pca_model, vgg_model = load_models()

    with st.spinner("Extracting features and predicting..."):
        features = vgg_model.predict(img_array)
        features = features.reshape(1, -1)
        features_pca = pca_model.transform(features)
        prediction = svm_model.predict(features_pca)
        probability = svm_model.predict_proba(features_pca)[0]

    st.subheader("📊 Prediction Results")
    predicted_class = classes[prediction[0]]
    st.success(f"**Predicted Stage: {predicted_class}**")

    st.subheader("📈 Confidence Scores")
    for i, (label, prob) in enumerate(zip(classes, probability)):
        st.write(f"**{label}:** {prob*100:.1f}%")

    fig, ax = plt.subplots()
    bars = ax.bar(classes, probability*100,
                  color=['green', 'orange', 'red', 'darkred'])
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Confidence")
    ax.set_ylim(0, 100)
    for bar, prob in zip(bars, probability):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    st.pyplot(fig)

st.info("💡 Model trained on your local dataset. Retrain with `python train_local.py` for updates.")
