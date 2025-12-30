import os
import json
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras
import tensorflow as tf
from werkzeug.utils import secure_filename
from utils.metadata_loader import load_bird_metadata

# ================= CONFIG =================
IMG_SIZE = 300
UPLOAD_FOLDER = "static/uploads"
CONFIDENCE_THRESHOLD = 0.50

# ================= LOAD CLASS NAMES =================
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)
print("✅ Loaded class names:", NUM_CLASSES)

# ================= LOAD METADATA =================
BIRD_METADATA = load_bird_metadata()
print("✅ Bird metadata loaded:", len(BIRD_METADATA))

# ================= FLASK APP =================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODELS =================
model_eff = keras.models.load_model("models/EfficientNetB7_best.keras")
model_dense = keras.models.load_model("models/DenseNet201_best.keras")
print("✅ Models loaded successfully")

# ================= IMAGE PREPROCESS =================
def preprocess_image(img_path):
    img = keras.preprocessing.image.load_img(
        img_path, target_size=(IMG_SIZE, IMG_SIZE)
    )
    arr = keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ================= ENSEMBLE PREDICTION =================
def predict_ensemble(img_path):
    img_arr = preprocess_image(img_path)

    eff_input = tf.keras.applications.efficientnet.preprocess_input(img_arr.copy())
    den_input = tf.keras.applications.densenet.preprocess_input(img_arr.copy())

    eff_pred = model_eff.predict(eff_input, verbose=0)
    den_pred = model_dense.predict(den_input, verbose=0)

    ensemble_pred = (eff_pred + den_pred) / 2.0

    idx = int(np.argmax(ensemble_pred))
    confidence = float(np.max(ensemble_pred))

    if confidence < CONFIDENCE_THRESHOLD or idx >= NUM_CLASSES:
        return "Unknown Bird Species", confidence * 100

    return CLASS_NAMES[idx], confidence * 100

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("images")

        image_results = []
        predicted_labels = []

        for file in files:
            if file.filename == "":
                continue

            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)

            label, conf = predict_ensemble(path)

            image_results.append({
                "image_path": path,
                "prediction": label,
                "confidence": f"{conf:.2f}"
            })

            if label != "Unknown Bird Species":
                predicted_labels.append(label)

        # ================= UNIQUE SPECIES =================
        unique_birds = sorted(set(predicted_labels))

        # ================= FETCH METADATA =================
        bird_details = {}
        for bird in unique_birds:
            bird_details[bird] = BIRD_METADATA.get(
                bird, {"error": "Scientific data not available"}
            )

        return render_template(
            "result.html",
            results=image_results,
            unique_birds=unique_birds,
            bird_details=bird_details,
            total_images=len(image_results)
        )

    return render_template("index.html")

# ================= RUN SERVER =================
if __name__ == "__main__":
    app.run(debug=True)
