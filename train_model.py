# train_model.py

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Settings
INIT_LR = 1e-4
EPOCHS = 10
BS = 32
DIRECTORY = "dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Loading and preprocessing images...")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        try:
            # Handle transparent images
            image = Image.open(img_path).convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # use alpha channel as mask
            image = background.resize((224, 224))
            image = np.array(image) / 255.0

            data.append(image)
            labels.append(category)

        except Exception as e:
            print(f"[WARNING] Skipping image {img_path}: {e}")

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Train-test split
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load MobileNetV2 base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Build the custom head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Final model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    epochs=EPOCHS
)

# Save the trained model
print("[INFO] Saving model to mask_detector.h5...")
model.save("mask_detector.h5")  # Keras 3-compatible format

# Save the label encoder
print("[INFO] Saving label encoder to label_encoder.pickle...")
with open("label_encoder.pickle", "wb") as f:
    pickle.dump(lb, f)

# Evaluate model
print("[INFO] Evaluating model...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# Print classification results
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Plot training accuracy and loss
print("[INFO] Plotting training performance...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="Train Acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
