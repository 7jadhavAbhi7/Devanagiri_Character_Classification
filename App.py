import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
import io
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from flask import render_template

app = Flask(__name__)

# Load your trained model
model = load_model("best_model.h5")  # Replace with your model file

# Character label mapping
class_labels = {0: 'yna', 1: 'taamatar', 2: 'thaa', 3: 'daa', 4: 'dhaa', 5: 'adna', 6: 'tabala', 7: 'tha', 8: 'da', 9: 'dha', 10: 'ka', 11: 'na', 12: 'pa', 13: 'pha', 14: 'ba', 15: 'bha', 16: 'ma', 17: 'yaw', 18: 'ra', 19: 'la', 20: 'waw', 21: 'kha', 22: 'motosaw', 23: 'petchiryakha', 24: 'patalosaw', 25: 'ha', 26: 'chhya', 27: 'tra', 28: 'gya', 29: 'ga', 30: 'gha', 31: 'kna', 32: 'cha', 33: 'chha', 34: 'ja', 35: 'jha', 36: '0', 37: '1', 38: '2', 39: '3', 40: '4', 41: '5', 42: '6', 43: '7', 44: '8', 45: '9'}
@app.route("/")
def index():
    return render_template("index.html")  # Serves the upload form

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    try:
        # Convert image to a format suitable for model prediction
        file = request.files['file']  # Read the image file from request
        image = Image.open(io.BytesIO(file.read()))  # Open the image with PIL

        # Convert PIL image to NumPy array (ensure RGB format)
        img = np.array(image)

        # Convert to OpenCV format (if needed)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize the image
        img_resized = cv2.resize(img, (32, 32))

        # Normalize and prepare for model input
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        pred = model.predict(img_array)
        predicted_class_index = np.argmax(pred, axis=1)[0]  # Get highest probability class
        predicted_label = class_labels.get(predicted_class_index, "Unknown")  # Get Marathi letter

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Vercel uses dynamic ports
    app.run(host="0.0.0.0", port=10000, debug=True)  # Run the ap
