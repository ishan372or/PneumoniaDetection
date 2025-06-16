from flask import Flask, render_template, request, url_for, redirect
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__, template_folder="template")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import os
import gdown

model_path = "model/pneumonia_model.h5"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=10dAbEJfVdJ2qtQ3--iC9NIR5f5Vq6HYN"  
    os.makedirs("model", exist_ok=True)
    gdown.download(url, model_path, quiet=False)

from tensorflow.keras.models import load_model
model = load_model(model_path)

# Define class labels (adjust if needed)
class_labels = ['Normal', 'Pneumonia']

# Initial load
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = url_for('static', filename=f'uploads/{filename}')
            return render_template('index.html', image_path=image_path)
    return redirect(url_for('index'))

# Predict button logic
@app.route('/predict', methods=['POST'])
def predict():
    image_path = request.form.get('image_path')
    if not image_path:
        return redirect(url_for('index'))

    # Convert relative URL to full local path
    local_image_path = image_path.replace('/static/', 'static/')

    # Preprocess the image
    img = image.load_img(local_image_path, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)[0][0]  # Binary classification
    predicted_label = class_labels[int(prediction > 0.5)]
    confidence = round(prediction * 100 if predicted_label == "Pneumonia" else (1 - prediction) * 100, 2)

    return render_template('index.html', image_path=image_path,
                           prediction=predicted_label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
