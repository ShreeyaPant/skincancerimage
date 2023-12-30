from flask import Flask, render_template, request,jsonify
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid Tkinter in non-main thread issues
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the trained model
skincancermodel = load_model("C:\\Users\\Asus\\Desktop\\important\\skinmodelepoch30final.hdf5")  # Update with the correct path

# Mapping for anatomical sites and gender
anatomical_site_mapping = ['head/neck', 'upper extremity', 'anterior torso', 'lower extremity', 'posterior torso', 'lateral torso', 'palms/soles', 'oral/genital', 'unknown']
gender_mapping = ['male', 'female', 'unknown']
class_labels = ['Benign Keratosis', 'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma']

# Severity information for each cancer type
severity_info = {
    'Benign Keratosis': 'Generally benign, but consult a dermatologist for evaluation.',
    'Melanoma': 'Potentially dangerous. Consult a dermatologist immediately.',
    'Basal Cell Carcinoma': 'Usually non-aggressive, but consult a dermatologist for proper evaluation.',
    'Squamous Cell Carcinoma': 'Can be aggressive. Consult a dermatologist promptly.'
}

def preprocess_image(uploaded_file):
    # If the input is a BytesIO object, directly open it as an Image
    if isinstance(uploaded_file, io.BytesIO):
        pil_image = Image.open(uploaded_file)
    else:
        # Otherwise, it's already a PIL Image
        pil_image = uploaded_file

    # Ensure the image is in RGB format
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Convert PIL Image to NumPy array
    img = np.array(pil_image)

    # Resize the image to (224, 224)
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

# Function to convert any image to JPG format
def convert_to_jpg(uploaded_file):
    pil_image = Image.open(uploaded_file)

    # Check if the image is already in JPG format
    if pil_image.format == "JPEG":
        return pil_image

    # Convert to JPG
    jpeg_buffer = io.BytesIO()
    pil_image.convert("RGB").save(jpeg_buffer, format="JPEG")
    return Image.open(jpeg_buffer)

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML template for the home page

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    jpg_image = convert_to_jpg(uploaded_file)

    age = int(request.form['age'])
    gender = request.form['gender']
    anatomical_site = request.form['anatomical_site']

    img = preprocess_image(jpg_image)

    gender_encoded = np.array([[1, 0, 0]]) if gender == 'male' else np.array([[0, 1, 0]]) if gender == 'female' else np.array([[0, 0, 1]])
    anatomical_site_encoded = np.zeros((1, 9))
    anatomical_site_encoded[0, anatomical_site_mapping.index(anatomical_site)] = 1

    age_array = np.array([[age]])

    prediction = skincancermodel.predict([img, age_array, gender_encoded, anatomical_site_encoded])

    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_labels[predicted_class_index]

    severity_info_text = severity_info.get(predicted_class_name, 'Severity information not available.')
    message = f"The predicted class is {predicted_class_name}. {severity_info_text}"

    # Create the 'static' directory if it doesn't exist
    save_directory = 'static'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    df = pd.DataFrame({'Class': class_labels, 'Probability': prediction.flatten()})
    # Plot the predicted probabilities as a bar chart using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(df['Class'], df['Probability'], color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Skin Cancer Classification Probability')
    
    # Save the chart as an image in the 'static' directory
    chart_path = os.path.join(save_directory, 'chart.png')
    plt.savefig(chart_path)
    plt.close()

    # Return JSON response
    return jsonify({
        'predicted_class': predicted_class_name,
        'message': message,
        'chart_url': chart_path,
    })

    return render_template('result.html', predicted_class=predicted_class_name, message=message)
if __name__ == '__main__':
    app.run(debug=True)