from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from tensorflow.keras.utils import register_keras_serializable
from flask_cors import CORS  # Import CORS for handling cross-origin requests

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Register the custom preprocessing function
@register_keras_serializable()
def effnet_preprocess(img):
    """
    Custom preprocessing function for EfficientNet.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img)

# Load the pre-trained model
def load_model():
    """
    Load the TensorFlow/Keras model.
    """
    model_path = 'Saved_Models/MoonArcModel.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'effnet_preprocess': effnet_preprocess}
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading the model: {str(e)}")
        raise RuntimeError(f"Error loading the model: {str(e)}")

# Initialize the model
try:
    model = load_model()
except RuntimeError as e:
    print("❌ Model loading failed. Exiting...")
    exit(1)

# Define class names
class_names = [
    'first quarter', 'full moon', 'new moon', 'no moon',
    'third quarter', 'waning crescent', 'waning gibbous',
    'waxing crescent', 'waxing gibbous'
]

def preprocess_image(image):
    """
    Preprocess the input image for prediction.
    """
    resized = cv2.resize(image, (224, 224))
    img_array = tf.keras.utils.img_to_array(resized)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests to predict moon phases.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(tf.nn.softmax(predictions[0])) * 100

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(float(confidence), 2)
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    return "MoonArc Backend is Running!", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Change to 8000
    app.run(debug=False, host='0.0.0.0', port=port)

