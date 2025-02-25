from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
from tensorflow.keras.utils import register_keras_serializable
from flask_cors import CORS  # Enable CORS for handling cross-origin requests
from werkzeug.exceptions import RequestEntityTooLarge

# Initialize Flask app
app = Flask(__name__)

# Set max request size to 10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": ["*", "https://moonarc.vercel.app", "http://localhost:3000"] }})

# Handle large file error
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'error': 'File too large. Max size is 10MB'}), 413

# Register the custom preprocessing function
@register_keras_serializable()
def resnet_preprocess(img):
    from tensorflow.keras.applications.resnet import preprocess_input
    return preprocess_input(img)

# Load the pre-trained model
def load_model():
    model_path = 'Saved_Models/Model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    try:
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'resnet_preprocess': resnet_preprocess}
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
except RuntimeError:
    print("❌ Model loading failed. Exiting...")
    exit(1)

# Define class names
class_names = [
    'first quarter', 'full moon', 'new moon', 'no moon',
    'third quarter', 'waning crescent', 'waning gibbous',
    'waxing crescent', 'waxing gibbous'
]

# Function to detect, crop, and enhance the moon in an image
def detect_and_crop_moon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        y_min = max(center[1] - radius, 0)
        y_max = min(center[1] + radius, image.shape[0])
        x_min = max(center[0] - radius, 0)
        x_max = min(center[0] + radius, image.shape[1])
        
        cropped = image[y_min:y_max, x_min:x_max]
        
        if cropped.size == 0:
            return image  # Return original if cropping fails
        
        lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((clahe_l, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    return image  # Return original if no moon detected

# Preprocess image for model prediction
def preprocess_image(image):
    image = detect_and_crop_moon(image)
    resized = cv2.resize(image, (224, 224))
    img_array = tf.keras.utils.img_to_array(resized)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
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
    return "🌙 MoonArc Backend is Running!", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # Running on port 8000
    app.run(debug=False, host='0.0.0.0', port=port)
