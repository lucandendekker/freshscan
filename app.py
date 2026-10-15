"""FreshScan: local inference with the original trained potato classifier."""

import io
import os
import zipfile
from collections import Counter
from pathlib import Path
from threading import Lock

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ("fresh", "rotten")
IMG_SIZE = (224, 224)


def load_model():
    """Load all saved weights, including the backbone, without network access."""
    import h5py
    from tf_keras.applications import MobileNetV2
    from tf_keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
    from tf_keras.models import Sequential

    # The original archive mixes keras and tf_keras layer metadata. Rebuild the
    # documented architecture, then restore EVERY saved inference variable.
    base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    model = Sequential([
        base, GlobalAveragePooling2D(), Dense(256, activation="relu"),
        BatchNormalization(), Dropout(0.5), Dense(2, activation="softmax"),
    ], name="cnn_v4_mobilenet_224")
    with zipfile.ZipFile(BASE_DIR / "cnn_v4_best.keras") as archive:
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
            restored = set()

            def restore(layer, path):
                group = weights[path]
                values = [group[str(index)][:] for index in range(len(group))]
                layer.set_weights(values)  # Fails on a missing or mismatched shape.
                restored.update(f"{path}/{index}" for index in range(len(group)))

            counters = Counter()
            names = {"Conv2D": "conv2d", "DepthwiseConv2D": "depthwise_conv2d", "BatchNormalization": "batch_normalization"}
            for layer in base.layers:
                if not layer.weights:
                    continue
                key = names[type(layer).__name__]
                index = counters[key]
                counters[key] += 1
                name = key if index == 0 else f"{key}_{index}"
                restore(layer, f"layers/functional/layers/{name}/vars")
            for layer, name in [(model.layers[2], "dense"), (model.layers[3], "batch_normalization"), (model.layers[5], "dense_1")]:
                restore(layer, f"layers/{name}/vars")

            saved = set()
            weights["layers"].visititems(lambda name, obj: saved.add(f"layers/{name}") if isinstance(obj, h5py.Dataset) else None)
            if restored != saved:
                raise ValueError("Not all saved model variables were restored")
    return model


def prepare_image(data):
    """Match the final training notebook: RGB, 224x224 and [0, 1] scaling."""
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(IMG_SIZE, Image.Resampling.NEAREST)
        return np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, 0)


def create_app(predictor=None):
    # No general static root: model files and source code must not be served.
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    model_lock = Lock()

    @app.get("/")
    def index():
        return send_from_directory(BASE_DIR, "index.html")

    @app.get("/potato_imgs/<path:filename>")
    def example_image(filename):
        return send_from_directory(BASE_DIR / "potato_imgs", filename)

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(error="Image is too large. Maximum upload size is 10 MB."), 413

    @app.post("/predict")
    def predict():
        nonlocal predictor
        file = request.files.get("file")
        if file is None or not file.filename:
            return jsonify(error="Choose an image first."), 400
        try:
            batch = prepare_image(file.read())
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="Could not read this image. Try a JPEG or PNG file."), 400

        try:
            with model_lock:
                if predictor is None:
                    predictor = load_model()
                probabilities = np.asarray(predictor.predict(batch, verbose=0))[0]
            return jsonify(
                **{"class": CLASS_NAMES[int(np.argmax(probabilities))]},
                fresh_confidence=round(float(probabilities[0]) * 100, 1),
                rotten_confidence=round(float(probabilities[1]) * 100, 1),
            )
        except Exception:
            app.logger.exception("Prediction failed")
            return jsonify(error="The classifier is unavailable. Check the server logs."), 500

    return app


app = create_app()

if __name__ == "__main__":
    print("Loading FreshScan model...")
    app = create_app(load_model())
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
