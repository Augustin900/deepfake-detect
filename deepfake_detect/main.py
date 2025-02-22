from requirements import *

# Load a pre-trained deepfake detection model (.keras format)
model = load_model('deepfake_detector.keras')  # Ensure you have a trained model saved in .keras format

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'mp4', 'jpg', 'png', 'wav'}

def preprocess_image(image_path):
    image = imread(image_path)
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize
    return expand_dims(image, axis=0)

def predict_deepfake(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]
    return "Deepfake" if prediction > 0.5 else "Real"

@app.route('/')
def index():
    return render_template('index.html', prediction=None, image_url=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part", image_url=None)
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file", image_url=None)
    if file and allowed_file(file.filename):
        filepath = join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        if file.filename.endswith(('.jpg', '.png')):
            prediction = predict_deepfake(filepath)
            return render_template('index.html', prediction=prediction, image_url=file.filename)
        
        return render_template('index.html', prediction="Uploaded and saved, but video analysis is not implemented.", image_url=None)

    return render_template('index.html', prediction="File type not allowed", image_url=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
