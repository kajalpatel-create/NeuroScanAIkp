from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import subprocess
import sys
import uuid
from werkzeug.utils import secure_filename


app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# LOAD CNN MODEL
# ============================================================

print("Loading CNN model...")

cnn_model = tf.keras.models.load_model(
    "models/brain_tumor_model.keras"
)

print("CNN model loaded successfully.")


# ============================================================
# LOAD MOBILENETV2 MODEL
# ============================================================

print("Loading MobileNetV2 model...")

mobilenet_model = tf.keras.models.load_model(
    "models/brain_tumor_mobilenetv2.keras"
)

print("MobileNetV2 model loaded successfully.")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

model_metrics = {
    "Test Accuracy": "84.31%",
    "Precision": "0.85",
    "Recall": "0.84",
    "F1-Score": "0.84"
}


model_comparison = {
    "CNN": {
        "Accuracy": "84.31%",
        "Precision": "0.85",
        "Recall": "0.84",
        "F1-Score": "0.84"
    },

    "MobileNetV2": {
        "Accuracy": "83.38%",
        "Precision": "0.84",
        "Recall": "0.83",
        "F1-Score": "0.83"
    }
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# IMAGE UPLOAD + AI ANALYSIS
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:

        return (
            "No image selected.",
            400
        )


    file = request.files["image"]


    if file.filename == "":

        return (
            "No image selected.",
            400
        )


    # --------------------------------------------------------
    # SECURE FILE NAME
    # --------------------------------------------------------

    original_filename = secure_filename(
        file.filename
    )


    # --------------------------------------------------------
    # UNIQUE FILE NAME
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex[:12]


    filename = (
        f"{unique_id}_"
        f"{original_filename}"
    )


    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    file.save(
        file_path
    )


    # ========================================================
    # STEP 1 — GRAD-CAM
    # ========================================================

    print("\n--------------------------------")
    print("Running Grad-CAM...")
    print("--------------------------------")


    gradcam_process = subprocess.run(
        [
            sys.executable,
            "gradcam.py",
            file_path
        ],
        capture_output=True,
        text=True
    )


    print(
        gradcam_process.stdout
    )


    if gradcam_process.stderr:

        print(
            "Grad-CAM message:",
            gradcam_process.stderr
        )


    gradcam_filename = (
        f"{os.path.splitext(filename)[0]}"
        "_gradcam.png"
    )


    gradcam_path = os.path.join(
        UPLOAD_FOLDER,
        gradcam_filename
    )


    gradcam_available = os.path.exists(
        gradcam_path
    )


    # ========================================================
    # STEP 2 — IMAGE PREPROCESSING FOR CNN
    # ========================================================

    image = Image.open(
        file_path
    ).convert("RGB")


    cnn_image = image.resize(
        (128, 128)
    )


    preprocessed_filename = (
        f"{os.path.splitext(filename)[0]}"
        "_preprocessed.png"
    )


    preprocessed_path = os.path.join(
        UPLOAD_FOLDER,
        preprocessed_filename
    )


    cnn_image.save(
        preprocessed_path
    )


    cnn_array = np.array(
        cnn_image,
        dtype=np.float32
    ) / 255.0


    cnn_array = np.expand_dims(
        cnn_array,
        axis=0
    )


    # ========================================================
    # STEP 3 — CNN PREDICTION
    # ========================================================

    print("\n--------------------------------")
    print("Running CNN prediction...")
    print("--------------------------------")


    cnn_prediction = cnn_model.predict(
        cnn_array,
        verbose=0
    )


    cnn_index = np.argmax(
        cnn_prediction[0]
    )


    cnn_class = class_names[
        cnn_index
    ]


    cnn_confidence = (
        float(
            cnn_prediction[
                0,
                cnn_index
            ]
        ) * 100
    )


    cnn_probabilities = {

        class_names[i]:

        round(
            float(
                cnn_prediction[
                    0
                ][i]
            ) * 100,
            2
        )

        for i in range(
            len(class_names)
        )
    }


    # ========================================================
    # STEP 4 — MOBILENETV2 PREPROCESSING
    # ========================================================

    mobile_image = image.resize(
        (160, 160)
    )


    mobile_array = np.array(
        mobile_image,
        dtype=np.float32
    ) / 255.0


    mobile_array = np.expand_dims(
        mobile_array,
        axis=0
    )


    # ========================================================
    # STEP 5 — MOBILENETV2 SECOND OPINION
    # ========================================================

    print("\n--------------------------------")
    print("Running MobileNetV2 prediction...")
    print("--------------------------------")


    mobile_prediction = (
        mobilenet_model.predict(
            mobile_array,
            verbose=0
        )
    )


    mobile_index = np.argmax(
        mobile_prediction[0]
    )


    mobile_class = class_names[
        mobile_index
    ]


    mobile_confidence = (
        float(
            mobile_prediction[
                0,
                mobile_index
            ]
        ) * 100
    )


    mobile_probabilities = {

        class_names[i]:

        round(
            float(
                mobile_prediction[
                    0
                ][i]
            ) * 100,
            2
        )

        for i in range(
            len(class_names)
        )
    }


    # ========================================================
    # STEP 6 — AI MODEL AGREEMENT
    # ========================================================

    if cnn_class == mobile_class:

        consensus_status = (
            "Models Agree"
        )

        consensus_message = (
            "Both AI models predicted "
            "the same class."
        )

        consensus = True

    else:

        consensus_status = (
            "Model Disagreement"
        )

        consensus_message = (
            "The two AI models produced "
            "different predictions. "
            "Further expert review is recommended."
        )

        consensus = False


    # ========================================================
    # STEP 7 — FINAL AI RESULT
    # ========================================================

    # We use CNN as the primary model because
    # it achieved higher independent test accuracy.

    prediction = cnn_class

    confidence = cnn_confidence

    probabilities = cnn_probabilities


    # ========================================================
    # STEP 8 — CONFIDENCE LEVEL
    # ========================================================

    if confidence >= 85:

        confidence_level = (
            "High Model Confidence"
        )

    elif confidence >= 60:

        confidence_level = (
            "Moderate Model Confidence"
        )

    else:

        confidence_level = (
            "Low Model Confidence"
        )


    # ========================================================
    # STEP 9 — ANALYSIS STATUS
    # ========================================================

    analysis_status = (
        "AI analysis completed successfully"
    )


    # ========================================================
    # STEP 10 — MODEL COMPARISON DATA
    # ========================================================

    model_results = {

        "CNN": {

            "prediction": cnn_class,

            "confidence": round(
                cnn_confidence,
                2
            )
        },

        "MobileNetV2": {

            "prediction": mobile_class,

            "confidence": round(
                mobile_confidence,
                2
            )
        }
    }


    # ========================================================
    # STEP 11 — RETURN RESULT PAGE
    # ========================================================
    return render_template(

        "result.html",

        # Original image
        filename=filename,

        original_filename=(
            original_filename
        ),


        # Preprocessed image
        preprocessed_filename=(
            "uploads/"
            + preprocessed_filename
        ),


        # Grad-CAM
        gradcam_filename=(
            "uploads/"
            + gradcam_filename
        ),

        gradcam_available=(
            gradcam_available
        ),


        # Final prediction
        prediction=prediction,

        confidence=confidence,

        confidence_level=(
            confidence_level
        ),


        # Probabilities
        probabilities=(
            probabilities
        ),


        # CNN result
        cnn_class=cnn_class,

        cnn_confidence=(
            cnn_confidence
        ),

        cnn_probabilities=(
            cnn_probabilities
        ),


        # MobileNetV2 result
        mobile_class=mobile_class,

        mobile_confidence=(
            mobile_confidence
        ),

        mobile_probabilities=(
            mobile_probabilities
        ),


        # Model agreement
        consensus=consensus,

        consensus_status=(
            consensus_status
        ),

        consensus_message=(
            consensus_message
        ),


        # Complete model result
        model_results=(
            model_results
        ),


        # Metrics
        model_metrics=(
            model_metrics
        ),

        model_comparison=(
            model_comparison
        ),


        # Status
        analysis_status=(
            analysis_status
        )
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )