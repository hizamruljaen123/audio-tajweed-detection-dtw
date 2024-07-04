from flask import Flask, jsonify, request
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import pandas as pd

app = Flask(__name__)

# Fungsi untuk ekstraksi fitur dari file audio
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        scaler = StandardScaler()
        mfccs = scaler.fit_transform(mfccs.T).T  # Normalisasi fitur
        return mfccs, y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

# Fungsi untuk menghitung jarak DTW antara dua fitur
def calculate_dtw_distance(feature1, feature2):
    distance, path = fastdtw(feature1.T, feature2.T, dist=euclidean)
    return distance, path

# Fungsi untuk memuat model dari file pickle
def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return [np.array(features) for features in model]
    except Exception as e:
        print(f"Error loading model {file_path}: {e}")
        return None

# Fungsi untuk deteksi Tajweed dan visualisasi
def detect_and_visualize_tajweed(file_path, models_folder):
    test_features, test_signal, sr = extract_features(file_path)
    if test_features is None or test_signal is None or sr is None:
        return None, None, None, None

    segment_length = 5  # Panjang segmen dalam detik
    segment_length_samples = segment_length * sr
    num_segments = len(test_signal) // segment_length_samples
    segment_results = []
    plot_urls = []

    models = {tajweed.split('_model')[0]: load_model(os.path.join(models_folder, tajweed))
              for tajweed in os.listdir(models_folder) if tajweed.endswith('_model.pkl')}

    for i in range(num_segments):
        start = i * segment_length_samples
        end = start + segment_length_samples
        segment = test_signal[start:end]
        segment_mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        scaler = StandardScaler()
        segment_mfcc = scaler.fit_transform(segment_mfcc.T).T  # Normalisasi fitur
        best_similarity = -np.inf
        best_tajweed = None

        for tajweed, tajweed_features in models.items():
            if tajweed_features is None:
                continue

            distances = [calculate_dtw_distance(train_feature, segment_mfcc)[0] for train_feature in tajweed_features]
            min_distance = min(distances)
            max_distance = max(distances)  # Menghitung nilai maksimum dari distances

            if max_distance == 0:
                similarity = 0  # Menghindari pembagian oleh nol
            else:
                similarity = max(0, 100 - (min_distance / max_distance * 100))  # Hitung similarity dalam persen

            if similarity > best_similarity:
                best_similarity = similarity
                best_tajweed = tajweed

        segment_results.append((i, best_tajweed, best_similarity))

        # Visualisasi segmen dengan Tajweed terdeteksi
        plt.figure(figsize=(8, 4))
        plt.plot(segment, label='Segment')
        plt.title(f'Segment {i}, Tajweed: {best_tajweed}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plot_file_path = f'plots/segment_{i}.png'  # Misalnya simpan gambar plot di folder plots
        plt.savefig(plot_file_path)
        plt.close()
        plot_urls.append(plot_file_path)

    return segment_results, plot_urls, test_signal, sr

# Route untuk deteksi Tajweed dan visualisasi dalam audio
@app.route('/detect_and_visualize_tajweed', methods=['POST'])
def api_detect_and_visualize_tajweed():
    file = request.files['file']
    if file:
        file_path = 'uploads/test_audio.opus'  # Misalnya simpan sementara di folder uploads
        file.save(file_path)
        models_folder = 'models'  # Path folder tempat model-model Tajweed disimpan
        segment_results, plot_urls, test_signal, sr = detect_and_visualize_tajweed(file_path, models_folder)
        if segment_results and plot_urls:
            return jsonify({"segment_results": segment_results, "plot_urls": plot_urls}), 200
        else:
            return jsonify({"message": "Error detecting and visualizing Tajweed."}), 500
    else:
        return jsonify({"message": "No file uploaded."}), 400

if __name__ == "__main__":
    app.run(debug=True)
