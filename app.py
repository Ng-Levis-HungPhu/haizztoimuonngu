from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import pickle

app = Flask(__name__)
CORS(app)

MODEL_DIR = "1. SAVING MODELS"

# Ánh xạ từng mode với thông tin cấu hình
MODE_CONFIG = {
    "NASA": {
        "features": ["lln", "ln", "swept", "mach", "aoa"],
        "model_prefix": "NASA"
    },
    "Von-Karman Nose": {
        "features": ["mach", "aoa"],
        "model_prefix": "Von-Karman Nose"
    },
    "Missile Shape 1": {
        "features": ["mach", "aoa"],
        "model_prefix": "Missile Shape 1"
    },
    "Missile Shape 2": {
        "features": ["mach", "aoa"],
        "model_prefix": "Missile Shape 2"
    },
    "Missile Shape 3": {
        "features": ["mach", "aoa"],
        "model_prefix": "Missile Shape 3"
    }
}


def load_model_and_scaler(prefix):
    """Load model CL, model CD và scaler dựa vào tiền tố model"""
    model_cl = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{prefix}_cl.h5"))
    model_cd = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{prefix}_cd.h5"))
    with open(os.path.join(MODEL_DIR, f"{prefix}.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model_cl, model_cd, scaler


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        mode = data.get("mode", "").strip()
        config = MODE_CONFIG.get(mode)

        if not config:
            return jsonify({"error": f"Unsupported mode: {mode}"}), 400

        # Kiểm tra và lấy thông số đầu vào
        try:
            input_values = [float(data[feature]) for feature in config["features"]]
        except (ValueError, KeyError):
            return jsonify({"error": "Vui lòng nhập đúng định dạng số cho tất cả các thông số"}), 400

        # Cảnh báo nếu mode là NASA và AOA ngoài vùng an toàn
        warning_msg = ""
        if mode == "NASA":
            aoa = float(data["aoa"])
            if not (-4 <= aoa <= 25):
                return jsonify({"error": "Góc tấn (Angle of Attack) phải nằm trong khoảng từ -4 đến 25"}), 400
            elif -4 <= aoa < -2.5:
                warning_msg = "Kết quả dự đoán có thể có sai số cao do AOA thấp bất thường."

        # Load model và scaler
        model_cl, model_cd, scaler = load_model_and_scaler(config["model_prefix"])

        input_array = np.array([input_values])
        input_scaled = scaler.transform(input_array)

        cl_pred = float(model_cl.predict(input_scaled, verbose=0)[0][0])
        cd_pred = float(model_cd.predict(input_scaled, verbose=0)[0][0])

        response = {
            "cl": round(cl_pred, 5),
            "cd": round(cd_pred, 5)
        }

        if warning_msg:
            response["warning"] = warning_msg

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
