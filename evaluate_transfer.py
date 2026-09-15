import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load MobileNetV2 model
model = tf.keras.models.load_model(
    "models/brain_tumor_mobilenetv2.keras"
)

test_dir = "dataset/brain_mri/Testing"

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("\nEvaluating MobileNetV2 model...\n")

# Model evaluation
loss, accuracy = model.evaluate(test_data, verbose=1)

print("\n===================================")
print("MOBILENETV2 EVALUATION RESULTS")
print("===================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

# Predictions
predictions = model.predict(test_data, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

# Classification report
print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# Confusion matrix
print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

# Plot confusion matrix
plt.figure(figsize=(7, 6))

plt.imshow(cm)
plt.title("NeuroScanAI - MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "mobilenetv2_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved as mobilenetv2_confusion_matrix.png")