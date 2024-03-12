from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image
from keras.utils import normalize

app = Flask(__name__)

# Define image size for preprocessing
INPUT_SIZE = 64

# Load the trained model
model = keras.models.load_model('BrainTumor10Epochs.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file and selected format from the POST request
        file = request.files['file']
        image_format = request.form['imageFormat']
        
        # Read the image file
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Preprocess the image based on the selected format
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        image_array = np.array(image)
        image_array = normalize(image_array, axis=1)
        image_array = image_array.reshape((1, INPUT_SIZE, INPUT_SIZE, 3))
        
        # Make prediction
        prediction = model.predict(image_array)
        
        # Convert prediction to string label
        result = "Yes Brain Tumor" if prediction[0][0] > 0.5 else "No Brain Tumor"
        
        # Return JSON response with the result
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
