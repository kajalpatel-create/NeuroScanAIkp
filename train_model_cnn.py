import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)
print("Dataset loading started...")
# Data Augmentation Visualization
augmentation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
train_dir = "dataset/brain_mri/Training"
test_dir = "dataset/brain_mri/Testing"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
class_mode="categorical",
    subset="training"
)
validation_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

print("Classes:", train_data.class_indices)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model training started...")

history = model.fit(
    train_data,
    epochs=5,

    validation_data=validation_data
)
import matplotlib.pyplot as plt
# Data Augmentation Visualization
augmentation_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
# Generate Augmented MRI Image

import os

os.makedirs("static", exist_ok=True)

augmented_data = augmentation_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode="categorical",
    shuffle=True
)

images, labels = next(augmented_data)

plt.figure(figsize=(5, 5))
plt.imshow(images[0])
plt.title("Augmented MRI Image")
plt.axis("off")

plt.savefig(
    "static/augmentation_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Augmentation visualization saved successfully.")
# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_graph.png", dpi=300)
plt.show()

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_graph.png", dpi=300)
plt.show()
model.save("models/brain_tumor_model.keras")

print("Model training completed!")
print("Model saved successfully.")