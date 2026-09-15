import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import sys
from PIL import Image


# ==========================================
# LOAD TRAINED CNN MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/brain_tumor_model.keras"
)

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ==========================================
# CHECK IMAGE PATH
# ==========================================

if len(sys.argv) < 2:
    print("No image path provided!")
    sys.exit()


img_path = sys.argv[1]

print("Selected image:", img_path)


# ==========================================
# LOAD IMAGE
# ==========================================

original_img = Image.open(
    img_path
).convert("RGB")

original_img = original_img.resize(
    (128, 128)
)


# ==========================================
# PREPROCESS IMAGE
# ==========================================

img_array = np.array(
    original_img,
    dtype=np.float32
) / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ==========================================
# FIND LAST CONVOLUTIONAL LAYER
# ==========================================

conv_layer = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):
        conv_layer = layer


if conv_layer is None:

    print(
        "No convolutional layer found!"
    )

    sys.exit()


print(
    "Grad-CAM layer:",
    conv_layer.name
)


# ==========================================
# CREATE GRAD-CAM MODEL
# ==========================================

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        conv_layer.output,
        model.output
    ]
)


# ==========================================
# CALCULATE GRADIENTS
# ==========================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = (
        grad_model(img_array)
    )

    predicted_index = tf.argmax(
        predictions[0]
    )

    class_output = predictions[
        0,
        predicted_index
    ]


# ==========================================
# PREDICTION
# ==========================================

predicted_class = class_names[
    int(predicted_index)
]

confidence = (
    float(
        predictions[
            0,
            predicted_index
        ]
    ) * 100
)


print(
    "Prediction:",
    predicted_class
)

print(
    f"Confidence: {confidence:.2f}%"
)


# ==========================================
# CALCULATE GRADIENTS
# ==========================================

grads = tape.gradient(
    class_output,
    conv_outputs
)


if grads is None:

    print(
        "Gradients could not be calculated!"
    )

    sys.exit()


# ==========================================
# GLOBAL AVERAGE POOLING
# ==========================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


conv_outputs = conv_outputs[0]


# ==========================================
# CREATE CLASS ACTIVATION MAP
# ==========================================

heatmap = tf.reduce_sum(
    conv_outputs *
    pooled_grads,
    axis=-1
)


# ==========================================
# APPLY RELU
# ==========================================

heatmap = tf.maximum(
    heatmap,
    0
)


# ==========================================
# NORMALIZE
# ==========================================

max_value = tf.reduce_max(
    heatmap
)


if max_value > 0:

    heatmap = (
        heatmap / max_value
    )


heatmap = heatmap.numpy()


# ==========================================
# RESIZE HEATMAP
# ==========================================

heatmap_img = Image.fromarray(
    np.uint8(
        heatmap * 255
    )
)

heatmap_img = heatmap_img.resize(
    (128, 128),
    Image.Resampling.BILINEAR
)

heatmap = (
    np.array(
        heatmap_img
    ) / 255.0
)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

output_folder = (
    "static/uploads"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ==========================================
# UNIQUE OUTPUT FILE
# ==========================================

image_name = os.path.splitext(
    os.path.basename(img_path)
)[0]

output_path = os.path.join(
    output_folder,
    f"{image_name}_gradcam.png"
)


# ==========================================
# CREATE PROFESSIONAL FIGURE
# ==========================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# ------------------------------------------
# ORIGINAL MRI
# ------------------------------------------

axes[0].imshow(
    original_img
)

axes[0].set_title(
    "Original MRI",
    fontsize=14
)

axes[0].axis("off")


# ------------------------------------------
# HEATMAP
# ------------------------------------------

heatmap_display = axes[1].imshow(
    heatmap,
    cmap="jet",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "AI Attention Map",
    fontsize=14
)

axes[1].axis("off")


# ------------------------------------------
# OVERLAY
# ------------------------------------------

axes[2].imshow(
    original_img
)

axes[2].imshow(
    heatmap,
    cmap="jet",
    alpha=0.5,
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "AI Focus Overlay",
    fontsize=14
)

axes[2].axis("off")


# ==========================================
# COLORBAR
# ==========================================

fig.colorbar(
    heatmap_display,
    ax=axes,
    fraction=0.025,
    pad=0.02,
    label="Activation"
)


# ==========================================
# MAIN TITLE
# ==========================================

fig.suptitle(
    "NeuroScanAI - Explainable AI Analysis\n"
    f"Prediction: {predicted_class} | "
    f"Confidence: {confidence:.2f}%",
    fontsize=16
)


plt.tight_layout(
    rect=[0, 0, 1, 0.90]
)


# ==========================================
# SAVE RESULT
# ==========================================

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# FINAL MESSAGE
# ==========================================

print(
    "\nGrad-CAM generated successfully!"
)

print(
    "Output:",
    output_path
)