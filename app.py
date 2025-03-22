from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load the trained model
model = load_model('best_model.keras')

# Load the precomputed features
features_dir = 'features'
precomputed_features = {}
for filename in os.listdir(features_dir):
    if filename.endswith('.npy'):
        product_id = filename.split('.')[0]
        feature_path = os.path.join(features_dir, filename)
        precomputed_features[product_id] = np.load(feature_path)

# Load the merged dataframe
merged_df = pd.read_csv('merged.csv')

# Ensure uploads directory exists
uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

# Set the path to the dataset images directory
dataset_images_dir = 'D:/Student projects hub/fashion-dataset/images'

# Preprocess the uploaded image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Find similar images
def find_similar_images(uploaded_image_path, top_n=5):
    uploaded_image = preprocess_image(uploaded_image_path)
    uploaded_feature = model.predict(uploaded_image)
    similarities = []
    for product_id, feature in precomputed_features.items():
        similarity = cosine_similarity(uploaded_feature, feature)
        similarities.append((product_id, similarity[0][0]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_images = similarities[:top_n]
    return similar_images

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(uploads_dir, filename)

@app.route('/dataset_images/<filename>')
def dataset_image_file(filename):
    return send_from_directory(dataset_images_dir, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filepath = os.path.join(uploads_dir, file.filename)
            file.save(filepath)
            similar_images = find_similar_images(filepath)
            results = []
            for product_id, similarity in similar_images:
                product_info = merged_df[merged_df['id'] == int(product_id)].iloc[0]
                results.append({
                    'id': product_id,
                    'filename': product_info['filename'],
                    'productDisplayName': product_info['productDisplayName'],
                    'similarity': similarity
                })
            return render_template('index.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)