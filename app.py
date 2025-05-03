from flask import Flask, request, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import time
from werkzeug.utils import secure_filename

# Flask configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Class labels and matches
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

fashion_matches = {
    'T-shirt/top': ['Trouser', 'Sneaker', 'Bag', 'Pullover'],
    'Trouser': ['Shirt', 'Ankle boot', 'T-shirt/top', 'Coat'],
    'Pullover': ['Trouser', 'Boot', 'Sneaker', 'Skirt'],
    'Dress': ['Sandal', 'Bag', 'Coat', 'Ankle boot'],
    'Coat': ['Trouser', 'Ankle boot', 'Dress', 'Pullover'],
    'Sandal': ['Dress', 'Bag', 'Skirt', 'T-shirt/top'],
    'Shirt': ['Trouser', 'Sneaker', 'Bag', 'Pullover'],
    'Sneaker': ['T-shirt/top', 'Trouser', 'Pullover', 'Skirt'],
    'Bag': ['Dress', 'Coat', 'Pullover', 'Shirt'],
    'Ankle boot': ['Coat', 'Trouser', 'Dress', 'Skirt']
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    img = Image.open(image_path).convert('L')  # Grayscale
    img = ImageOps.invert(img)  # Invert colors (white on black)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28), img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    matches = []
    error = None
    image_url = None
    processed_url = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            error = "No file selected"
        elif not allowed_file(file.filename):
            error = "Unsupported file type"
        else:
            filename = secure_filename(file.filename)
            unique_filename = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            image_url = url_for('static', filename=f'uploads/{unique_filename}')

            try:
                img_array, processed_img = prepare_image(filepath)

                # Save processed preview
                processed_filename = f"processed_{unique_filename}"
                processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                processed_img.save(processed_path)
                processed_url = url_for('static', filename=f'uploads/{processed_filename}')

                prediction_array = model.predict(img_array)
                label = class_names[np.argmax(prediction_array)]
                prediction = label
                matches = fashion_matches.get(label, [])

            except Exception as e:
                error = f"Prediction failed: {e}"

    return render_template("index.html",
                           prediction=prediction,
                           matches=matches,
                           error=error,
                           image_url=image_url,
                           processed_url=processed_url)

if __name__ == '__main__':
    app.run(debug=True)
