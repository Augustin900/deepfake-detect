import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer           # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences   # type: ignore
from tensorflow.keras.models import Sequential                      # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
from flask import *
import os

# Step 1: Load the dataset
def load_dataset():
    # Replace with your own dataset or path to a CSV file
    data = {
        'text': [
            "The earth is flat and NASA is lying.",
            "The new economic policies have boosted the GDP significantly.",
            "Aliens built the pyramids according to secret documents.",
            "Scientists discover a new cure for cancer.",
        ],
        'label': [1, 0, 1, 0]  # 1 = fake, 0 = real
    }
    return pd.DataFrame(data)

data = load_dataset()

# Step 2: Preprocess the dataset
texts = data['text'].values
labels = data['label'].values

# Encode labels
label_encoder = LabelBinarizer()
labels = label_encoder.fit_transform(labels)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize and pad text
max_vocab_size = 5000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

# Step 3: Build the model
model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=64, input_length=max_sequence_length),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 4: Train the model
history = model.fit(
    X_train_padded, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_padded, y_test),
    verbose=1
)

# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Step 6: Predict on new data
def predict_fake_news(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    return "Fake News" if prediction > 0.5 else "Real News"

# Flask app for uploading and predicting
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'mp4', 'jpg', 'png', 'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        if file.filename.endswith('.txt'):
            with open(filepath, 'r') as f:
                text = f.read()
            prediction = predict_fake_news(text)
            return render_template('index.html', prediction=prediction, image_url=None)
        
        if file.filename.endswith(('.jpg', '.png')):
            return render_template('index.html', prediction=None, image_url=file.filename)

        # Add logic for videos, audio if needed
        return render_template('index.html', prediction="Uploaded and saved, but further processing not implemented.", image_url=None)

    return render_template('index.html', prediction="File type not allowed", image_url=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
