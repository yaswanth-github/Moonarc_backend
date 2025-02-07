from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging
from tensorflow.keras.utils import register_keras_serializable
from flask_cors import CORS  # Import CORS for handling cross-origin requests

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Register the custom preprocessing function
@register_keras_serializable()
def effnet_preprocess(img):
    """
    Custom preprocessing function for EfficientNet.
    This function is registered with Keras to ensure it can be serialized/deserialized.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img)

# Load the pre-trained model
def load_model():
    """
    Load the TensorFlow/Keras model.
    Ensure the custom function `effnet_preprocess` is available during loading.
    """
    try:
        model_path = os.path.join(os.getcwd(), 'Saved_Models/MoonArc89.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logging.info("Loading model...")
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
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {str(e)}")
        raise RuntimeError(f"Error loading the model: {str(e)}")

# Initialize the model
try:
    model = load_model()
except RuntimeError as e:
    logging.error("Failed to load model. Exiting application.")
    exit(1)

# Define class names (based on the dataset structure in the PDF)
class_names = [
    'first quarter', 'full moon', 'new moon', 'no moon',
    'third quarter', 'waning crescent', 'waning gibbous',
    'waxing crescent', 'waxing gibbous'
]

def preprocess_image(image):
    """
    Preprocess the input image for prediction.
    Resizes the image to (224, 224) and converts it into a batched tensor.
    """
    resized = cv2.resize(image, (224, 224))  # Resize to match model input size
    img_array = tf.keras.utils.img_to_array(resized)  # Convert to NumPy array
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests to predict moon phases.
    Expects an image file in the request.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))  # Read image from file
        # Ensure the image has 3 channels (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert to RGB if necessary
        image = np.array(image)  # Convert to NumPy array

        # Preprocess the image for the model
        processed_image = preprocess_image(image)

        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]  # Get predicted class
        confidence = np.max(tf.nn.softmax(predictions[0])) * 100  # Get confidence score

        # Return the result as JSON
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        # Handle unexpected errors gracefully
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    return "MoonArc Backend is Running!", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Use Render-assigned port or default 5001
    app.run(debug=False, host='0.0.0.0', port=port)
