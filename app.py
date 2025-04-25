from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('ensemble_model.h5')
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

remedies = {
    'No_DR': "No signs of Diabetic Retinopathy. Keep managing your blood sugar levels and schedule regular eye checkups.",
    'Mild': "Mild non-proliferative DR. Regular monitoring, controlling blood sugar, blood pressure, and cholesterol are advised.",
    'Moderate': "Moderate non-proliferative DR. Regular eye exams and lifestyle management are critical. Your doctor may prescribe medication.",
    'Severe': "Severe non-proliferative DR. More frequent monitoring and treatments like anti-VEGF injections may be recommended.",
    'Proliferate_DR': "Proliferative DR. This is an advanced stage that may require laser treatments, injections, or surgery. Consult a retinal specialist immediately."
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    remedy = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 224, 224, 3)

            pred = model.predict(img_array)
            predicted_class = class_names[np.argmax(pred)]
            prediction = f"Predicted Class: {predicted_class}"
            remedy = remedies[predicted_class]
            image_path = filepath

            return render_template('index.html', prediction=prediction, remedy=remedy, image_path=os.path.basename(filepath))

    return render_template('index.html', prediction=prediction, remedy=remedy, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
