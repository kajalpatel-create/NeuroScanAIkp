import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/brain_tumor_model.keras"
)


# ==========================================
# TEST DATASET
# ==========================================

test_dir = "dataset/brain_mri/Testing"

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


print("\nEvaluating model...\n")


# ==========================================
# MODEL EVALUATION
# ==========================================

loss, accuracy = model.evaluate(
    test_data,
    verbose=1
)

print("\n===================================")
print("MODEL EVALUATION RESULTS")
print("===================================")

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(
    test_data,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_data.classes


# ==========================================
# CLASS NAMES
# ==========================================

class_names = list(
    test_data.class_indices.keys()
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

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


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ==========================================
# PLOT CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "NeuroScanAI - Confusion Matrix"
)

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


# Display values inside matrix

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
    "confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved as confusion_matrix.png"
)


# ==========================================
# ROC CURVE AND AUC
# ==========================================

y_true_binary = label_binarize(
    y_true,
    classes=range(len(class_names))
)


plt.figure(figsize=(8, 6))


for i in range(len(class_names)):

    fpr, tpr, _ = roc_curve(
        y_true_binary[:, i],
        predictions[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    print(
        f"{class_names[i]} AUC: {roc_auc:.4f}"
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_names[i]} (AUC = {roc_auc:.2f})"
    )


# Diagonal reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.title(
    "NeuroScanAI - ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.grid()

plt.tight_layout()


plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()


print(
    "\nROC Curve saved as roc_curve.png"
)