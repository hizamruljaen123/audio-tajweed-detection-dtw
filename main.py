import os
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import soundfile as sf

def extract_features(file_path):
    y, sr = sf.read(file_path)  # Menggunakan soundfile untuk membaca file .opus
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs

def calculate_dtw_distance(feature1, feature2):
    distance, path = fastdtw(feature1.T, feature2.T, dist=euclidean)
    return distance

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Ekstraksi fitur untuk semua file audio latih
train_audio_folder = 'data_latih'  # Ganti dengan path folder yang berisi file .opus
train_files = [os.path.join(train_audio_folder, f) for f in os.listdir(train_audio_folder) if f.endswith('.opus')]

train_features = [extract_features(file) for file in train_files]

# Simpan model
save_model(train_features, 'train_features.pkl')

# Muat model
loaded_train_features = load_model('train_features.pkl')

# Ekstraksi fitur untuk data uji
test_audio_path = 'test.opus'  # Ganti dengan path file audio uji
test_features = extract_features(test_audio_path)

# Hitung jarak DTW antara data uji dan semua fitur data latih
distances = [calculate_dtw_distance(train_feature, test_features) for train_feature in loaded_train_features]

# Menampilkan hasil jarak DTW dan menghitung persentase kemiripan
min_distance = min(distances)
max_distance = max(distances)

for i, distance in enumerate(distances):
    similarity = 100 - ((distance - min_distance) / (max_distance - min_distance) * 100)
    print(f"DTW Distance with train audio {i+1}: {distance}, Similarity: {similarity:.2f}%")
