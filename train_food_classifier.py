import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET = "food_subset"
IMG_SIZE = 128


def extract_features(img):
    """Extract HOG + color histogram features from an image."""
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # HOG features (shape & texture)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Color histograms in HSV (color distribution)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    color_features = color_features / (color_features.sum() + 1e-7)  # normalize

    return np.concatenate([hog_features, color_features])


X = []
y = []

for category in os.listdir(DATASET):
    path = os.path.join(DATASET, category)
    if not os.path.isdir(path):
        continue

    images = os.listdir(path)
    print(f"  Loading {len(images)} images from '{category}'...")

    for image_name in images:
        img_path = os.path.join(path, image_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_features(img)
            X.append(features)
            y.append(category)
        except Exception as e:
            print(f"    Skipping {image_name}: {e}")

print(f"\nTotal samples: {len(X)}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\nAccuracy: {acc:.4f} ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/food_classifier.pkl")
print("Food classifier saved to models/food_classifier.pkl")