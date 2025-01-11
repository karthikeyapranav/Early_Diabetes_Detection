from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("models/diabetes_model_final.h5")

# Class labels dictionary
class_labels = {
    0: 'Mild Diabetic Retinopathy detected',
    1: 'Moderate Diabetic Retinopathy detected',
    2: 'No Diabetic Retinopathy detected',
    3: 'Proliferative Diabetic Retinopathy detected',
    4: 'Severe Diabetic Retinopathy detected'
}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_precautions(diagnosis):
    precautions = {
        'Mild Diabetic Retinopathy detected': "Monitor your blood sugar levels regularly. Maintain a healthy diet and exercise."
        "Explanation Blood sugar High blood sugar levels can damage the retina, the part of the eye that detects light. Monitoring blood sugar levels and taking insulin or other diabetes medications can help prevent diabetic retinopathy. Diet Eating a healthy, balanced diet can help manage diabetic retinopathy. This includes eating fruits and vegetables, whole grains, and plant-based proteins. ExerciseRegular physical activity can help manage diabetic retinopathy. This includes getting at least 150 minutes of moderate aerobic activity each week",
        'Moderate Diabetic Retinopathy detected': "Consult a doctor for proper treatment. Maintain a healthy lifestyle and follow prescribed medications. TreatmentRegular monitoring: In the early stages, your eye doctor may monitor your eyes closely. Injections: Anti-VEGF drugs or corticosteroids can help slow down or reverse the disease. Laser treatment: A laser beam can shrink abnormal blood vessels and stop them from leaking. Eye surgery: If the disease is advanced, your eye doctor may recommend a vitrectomy to remove blood or scar tissue from the eye.",
        'No Diabetic Retinopathy detected': "No diabetic retinopathy detected. Keep monitoring blood sugar levels and maintain a healthy lifestyle.Managing your blood sugar levels is the best way to prevent diabetic retinopathy. Controlling your blood sugar can also help prevent it from getting worse. If your blood sugar levels are well-controlled, you can delay the onset of diabetic retinopathy and prevent vision loss",
        'Proliferative Diabetic Retinopathy detected': "Seek immediate medical attention. Follow the prescribed treatment plan and consult an eye specialist.Laser treatment: A doctor uses a laser to shrink the abnormal blood vessels and seal leaks. This treatment can stop or slow the leakage of blood and fluid. Eye injections: Used to treat severe maculopathy Steroid eye implants: Used to treat severe maculopathy if injections don't work Eye surgery: Used to remove blood or scar tissue from the eye if laser treatment isn't possible ",
        'Severe Diabetic Retinopathy detected': "Seek immediate medical attention. Severe diabetic retinopathy requires urgent treatment to avoid vision loss.Treatments for diabetic retinopathy include:Laser treatment: Can treat abnormal blood vessel growth and leaking blood vessels.Injections: Medications can be injected into the eye to slow abnormal blood vessel growth and treat macular edema. Vitrectomy: An outpatient surgery that involves removing the vitreous, the jelly-like fluid that fills the center of the eye. Eye surgery: Can remove blood or scar tissue from the eye. "
    }
    return precautions.get(diagnosis, "No precautions available.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = prepare_image(filepath)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            diagnosis = class_labels[predicted_class]
            precautions = get_precautions(diagnosis)

            return jsonify({
                'prediction': diagnosis,
                'precautions': precautions
            })
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
