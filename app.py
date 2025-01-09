import os
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("D:/Diabetes-2/models/diabetes_model_final.h5")

# Class labels dictionary
class_labels = {0: 'mild', 1: 'moderate', 2: 'no_dr', 3: 'proliferate_dr', 4: 'sever'}

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Prepare the image for prediction
        image = prepare_image(filepath)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        diagnosis = class_labels[predicted_class]
        
        precautions = get_precautions(diagnosis)
        
        return jsonify({
            'prediction': diagnosis,
            'precautions': precautions
        })
    else:
        return jsonify({"error": "Invalid file format. Only PNG, JPG, JPEG allowed."}), 400

def get_precautions(diagnosis):
    precautions_dict = {
        'mild': "Monitor your blood sugar levels regularly. Maintain a healthy diet and exercise regularly.",
        'moderate': "Consult a doctor for proper treatment. Maintain a healthy lifestyle and follow prescribed medications.",
        'no_dr': "No diabetic retinopathy detected. Keep monitoring blood sugar levels regularly.",
        'proliferate_dr': "Seek immediate medical attention. Follow the prescribed treatment plan and consult an eye specialist.",
        'sever': "Seek immediate medical attention. Severe diabetic retinopathy requires urgent treatment to avoid vision loss."
    }
    return precautions_dict.get(diagnosis, "No precautions available.")

if __name__ == '__main__':
    app.run(debug=True)
