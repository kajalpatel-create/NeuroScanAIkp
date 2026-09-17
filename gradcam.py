import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import sys
from PIL import Image


# ==========================================
# LOAD MODEL
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
# IMAGE PATH
# ==========================================

if len(sys.argv) < 2:
    print("No image path provided!")
    sys.exit(1)

img_path = sys.argv[1]

print("--------------------------------")
print("Running Grad-CAM...")
print("--------------------------------")
print("Selected image:", img_path)


if not os.path.exists(img_path):
    print("Image file not found!")
    sys.exit(1)


# ==========================================
# LOAD IMAGE
# ==========================================

original_img = Image.open(
    img_path
).convert("RGB")

original_img = original_img.resize(
    (128, 128)
)

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

conv_layer = model.get_layer(
    "conv2d_1"
)

print(
    "Grad-CAM layer:",
    conv_layer.name
)


# ==========================================
# MANUAL FORWARD PASS
# ==========================================

with tf.GradientTape() as tape:

    x = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32
    )

    # First convolution
    x = model.get_layer(
        "conv2d"
    )(x)

    # First pooling
    x = model.get_layer(
        "max_pooling2d"
    )(x)

    # Target convolution
    conv_output = model.get_layer(
        "conv2d_1"
    )(x)

    # Watch actual convolution output
    tape.watch(conv_output)

    # Second pooling
    x = model.get_layer(
        "max_pooling2d_1"
    )(conv_output)

    # Flatten
    x = model.get_layer(
        "flatten"
    )(x)

    # Dense
    x = model.get_layer(
        "dense"
    )(x)

    # Dropout - inference mode
    x = model.get_layer(
        "dropout"
    )(x, training=False)

    # Final output
    predictions = model.get_layer(
        "dense_1"
    )(x)

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
# GRADIENT
# ==========================================

grads = tape.gradient(
    class_output,
    conv_output
)


if grads is None:
    print(
        "Gradients could not be calculated!"
    )
    sys.exit(1)


print("Gradients calculated successfully!")


# ==========================================
# GLOBAL AVERAGE POOLING
# ==========================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_output = conv_output[0]


# ==========================================
# CREATE HEATMAP
# ==========================================

heatmap = tf.reduce_sum(
    conv_output * pooled_grads,
    axis=-1
)

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

if float(max_value) > 0:
    heatmap = heatmap / max_value

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
# OUTPUT
# ==========================================

output_folder = "static/uploads"

os.makedirs(
    output_folder,
    exist_ok=True
)

image_name = os.path.splitext(
    os.path.basename(img_path)
)[0]

output_path = os.path.join(
    output_folder,
    f"{image_name}_gradcam.png"
)


# ==========================================
# FIGURE
# ==========================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(
    original_img
)

axes[0].set_title(
    "Original MRI",
    fontsize=14
)

axes[0].axis("off")


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


fig.colorbar(
    heatmap_display,
    ax=axes,
    fraction=0.025,
    pad=0.02,
    label="Activation"
)


fig.suptitle(
    "NeuroScanAI - Explainable AI Analysis\n"
    f"Prediction: {predicted_class} | "
    f"Confidence: {confidence:.2f}%",
    fontsize=16
)


plt.tight_layout(
    rect=[0, 0, 1, 0.90]
)


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("--------------------------------")
print("Grad-CAM generated successfully!")
print("Output:", output_path)
print("--------------------------------")