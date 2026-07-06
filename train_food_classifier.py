import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ==========================================
# 1. SETUP DATA LOCATIONS
# ==========================================
# Replace these with the actual paths to your training and validation image folders
TRAIN_DIR = "food_subset"

# ==========================================
# 2. IMAGE PREPROCESSING (IMAGE GENERATORS)
# ==========================================
# MobileNetV2 requires images scaled between -1 and 1, and sized to 224x224
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalizes pixel values
    validation_split=0.2     # Uses 20% of your data for testing/accuracy verification
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 3. BUILD THE DEEP LEARNING MODEL
# ==========================================
print("Loading pre-trained MobileNetV2 base...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the pre-trained weights

# Add your custom food classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x) # 101 food categories

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. TRAIN THE MODEL (UNCOMMENTED & READY)
# ==========================================
print("Starting Deep Learning Training...")
model.fit(
    train_generator,
    epochs=5,  # You can increase this to 10 for better accuracy if your PC has a GPU
    validation_data=validation_generator
)

# ==========================================
# 5. SAVE THE MODEL WEIGHTS
# ==========================================
os.makedirs("models", exist_ok=True)
# Save as a native Keras/TensorFlow model format (recommended for deep learning)
model.save("models/food_classifier.h5")
print("Successfully saved Deep Learning model to models/food_classifier.h5!")