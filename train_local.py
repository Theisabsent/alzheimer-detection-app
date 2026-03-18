import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Local paths (relative to current script dir which will be alzheimer-detection-app)
BASE_DIR = '..'
IMG_SIZE = 224
classes = {
    "Nondemented": 0,
    "Verymilddemented": 1,
    "MildDemented": 2,
    "Moderatedemented": 3
}

X, y = [], []
print("Loading images from local folders...")

for class_name, label in classes.items():
    class_folder = os.path.join(BASE_DIR, class_name)
    if not os.path.exists(class_folder):
        print(f"Warning: {class_folder} not found. Skipping.")
        continue
    image_files = [f for f in os.listdir(
        class_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images in {class_name}")
    for file in image_files:
        img_path = os.path.join(class_folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        X.append(img)
        y.append(label)

X = np.array(X, dtype="float32")
y = np.array(y)
print(f"Total images loaded: {len(X)} with labels: {np.bincount(y)}")

if len(X) == 0:
    raise ValueError("No images loaded! Check folder names and images.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

print("Extracting VGG16 features...")
base_model = VGG16(weights="imagenet", include_top=False,
                   input_shape=(224, 224, 3))
cnn_model = Model(inputs=base_model.input, outputs=base_model.output)

train_features = cnn_model.predict(X_train, verbose=1)
test_features = cnn_model.predict(X_test, verbose=1)
train_features = train_features.reshape(train_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)

print("Applying PCA...")
pca = PCA(n_components=min(300, train_features.shape[1]))
train_pca = pca.fit_transform(train_features)
test_pca = pca.transform(test_features)
print(
    f"PCA reduced to {train_pca.shape[1]} components. Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

print("Training SVM...")
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(train_pca, y_train)

y_pred = svm.predict(test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(classes.keys())))

joblib.dump(svm, "svm_local.pkl")
joblib.dump(pca, "pca_local.pkl")
print("\nModels saved: svm_local.pkl, pca_local.pkl")
print("Training complete! You can now run the Streamlit app.")
