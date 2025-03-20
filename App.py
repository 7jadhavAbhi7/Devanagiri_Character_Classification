from flask import Flask, request, jsonify, render_template
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your trained model
model = load_model("best_model.keras")  # Replace with your model file

# Character label mapping
class_labels = class_labels = ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
                'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
                'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
                'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', '०', '१', '२', 
                '३', '४', '५', '६', '७', '८', '९']
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
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((32, 32))  # Ensure the correct input size
        image = image.convert("RGB")  # Ensure 3 channels
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(image)
        predicted_class_idx = np.argmax(predictions)
        predicted_label = class_labels[predicted_class_idx]

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
