from flask import Flask, jsonify, request, send_file, render_template, send_from_directory
from flask_cors import CORS  # Import Flask-CORS
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import pandas as pd
import uuid  # Untuk menghasilkan nama file yang unik
from ftplib import FTP
from pydub import AudioSegment
import tempfile

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua rute

# Konfigurasi FTP
FTP_HOST = '103.147.154.185'
FTP_PORT = 21
FTP_USER = 'main_data@hjmainserver.my.id'
FTP_PASSWORD = 'komputer123@@'

# Fungsi untuk mengirim file ke FTP
def send_to_ftp(file_path, remote_filename, remote_directory):
    try:
        with FTP() as ftp:
            ftp.connect(FTP_HOST, FTP_PORT)
            ftp.login(FTP_USER, FTP_PASSWORD)
            ftp.cwd(remote_directory)  # Ganti dengan folder tujuan di FTP
            
            with open(file_path, 'rb') as f:
                ftp.storbinary(f'STOR {remote_filename}', f)
            
            print(f"File {remote_filename} berhasil diunggah ke FTP.")
    except Exception as e:
        print(f"Error uploading file to FTP: {e}")

# Fungsi untuk memastikan folder uploads ada dan menyimpan file di sana
def save_uploaded_file(file):
    uploads_dir = 'static/uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    # Dapatkan ekstensi file
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(uploads_dir, f'{uuid.uuid4().hex}{file_extension}')
    
    # Simpan file audio sesuai format yang didukung oleh pydub
    supported_formats = ['.wav', '.mp3', '.opus']
    if file_extension not in supported_formats:
        return None  # Jika format tidak didukung, kembalikan None
    
    # Konversi ke WAV jika bukan WAV
    if file_extension != '.wav':
        audio = AudioSegment.from_file(file)
        audio.export(file_path, format="wav")
    else:
        file.save(file_path)
    
    return file_path

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

        print(f"Processing segment {i}...")
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

            print(f"    Matching with model '{tajweed}': Similarity {similarity:.2f}%")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_tajweed = tajweed

        segment_results.append((i, best_tajweed, best_similarity))

    # Gabungkan hasil deteksi ke dalam satu grafik
    plt.figure(figsize=(12, 6))
    plt.plot(test_signal, label='Test Signal', color='gray', alpha=0.5)
    colors = plt.get_cmap('tab20', len(models))

    for i, (segment_index, tajweed, _) in enumerate(segment_results):
        start = segment_index * segment_length_samples
        end = start + segment_length_samples
        if tajweed is not None:
            plt.plot(range(start, end), test_signal[start:end], color=colors(i % len(models)), label=tajweed)

    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Combined Tajweed Detection in Test Audio')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Simpan grafik ke folder static/result dengan nama unik
    result_folder = 'static/result'
    os.makedirs(result_folder, exist_ok=True)
    plot_file_name = f'{uuid.uuid4().hex}.png'
    plot_file_path = os.path.join(result_folder, plot_file_name)
    plt.savefig(plot_file_path)
    plt.close()

    # Mengembalikan hasil deteksi dan nama file plot
    plot_urls.append(plot_file_name)
    return segment_results, plot_urls, test_signal, sr

# Route untuk unggah file audio, deteksi Tajweed, dan visualisasi
@app.route('/upload_detect_visualize', methods=['POST'])
def upload_detect_visualize_tajweed():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file."}), 400

    file_path = save_uploaded_file(file)
    if file_path is None:
        return jsonify({"message": "Unsupported file format."}), 400

    models_folder = '../models'  # Path folder tempat model-model Tajweed disimpan

    segment_results, plot_urls, _, _ = detect_and_visualize_tajweed(file_path, models_folder)

    if segment_results and plot_urls:
        # Mengirim file audio yang diunggah ke FTP
        audio_filename = os.path.basename(file_path)
        audio_remote_dir = 'public_html/tahsin/upload/audio_upload'
        send_to_ftp(file_path, audio_filename, audio_remote_dir)

        # Mengirim file gambar plot ke FTP
        for plot_url in plot_urls:
            plot_file_path = os.path.join('static/result', plot_url.split('/')[-1])  # Mendapatkan path lokal file plot
            plot_remote_filename = plot_url.split('/')[-1]
            plot_remote_dir = 'public_html/tahsin/upload/result_chart'
            send_to_ftp(plot_file_path, plot_remote_filename, plot_remote_dir)

        # Mengembalikan JSON hasil deteksi, URL plot, dan URL file audio serta file audio yang diunggah
        results = {
            "segment_results": segment_results,
            "plot_urls": [f"https://hjmainserver.my.id/tahsin/upload/result_chart/{url}" for url in plot_urls],
            "audio_file": f"https://hjmainserver.my.id/tahsin/upload/audio_upload/{audio_filename}"
        }
        print(f"Detection results: {results}")
        return jsonify(results), 200
    else:
        return jsonify({"message": "Error detecting and visualizing Tajweed."}), 500

# Fungsi untuk mengakses file plot dari direktori static/result
@app.route('/result_chart/<path:filename>')
def serve_plot(filename):
    return send_from_directory('static/result', filename)

if __name__ == '__main__':
    app.run(debug=True)
