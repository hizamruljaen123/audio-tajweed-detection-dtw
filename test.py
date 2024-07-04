import os
import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None, None, None
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        scaler = StandardScaler()
        mfccs = scaler.fit_transform(mfccs.T).T  # Normalisasi fitur
        return mfccs, y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

def save_extracted_features(features, file_path):
    np.savetxt(file_path, features, delimiter=',')
    print(f"Extracted features saved to {file_path}")

def calculate_dtw_distance(feature1, feature2):
    distance, path = fastdtw(feature1.T, feature2.T, dist=euclidean)
    return distance, path

def load_model(file_path):
    if not os.path.exists(file_path):
        print(f"Model file {file_path} does not exist.")
        return None
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return [np.array(features) for features in model]
    except Exception as e:
        print(f"Error loading model {file_path}: {e}")
        return None

def detect_tajweed(test_audio_path, models_folder, segment_length=5):
    test_features, test_signal, sr = extract_features(test_audio_path)
    if test_features is None or test_signal is None or sr is None:
        print("Error extracting features from test audio.")
        return [], None, None, None

    save_extracted_features(test_features, "test_features.txt")
    segment_length_samples = segment_length * sr
    num_segments = len(test_signal) // segment_length_samples
    segment_results = []

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

            print(f"Segment {i}, Tajweed {tajweed}, Similarity: {similarity:.2f}%")
            if similarity > best_similarity:
                best_similarity = similarity
                best_tajweed = tajweed

        segment_results.append((i, best_tajweed, best_similarity))

    print("Pattern detection completed.")
    return segment_results, test_signal, sr, segment_length_samples

def plot_audio_signal(test_signal, sr):
    plt.figure(figsize=(14, 7))
    plt.plot(test_signal, label='Audio Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal')
    plt.legend(loc='upper right')
    plt.show()

def plot_detected_tajweed_segments(test_signal, segment_results, sr, segment_length_samples, tajweeds):
    colors = plt.get_cmap('tab20', len(tajweeds))
    plt.figure(figsize=(14, 7))
    plt.plot(test_signal, label='Test Signal', color='gray', alpha=0.5)
    
    legend_labels = set()  # Set untuk menyimpan label legenda unik

    for i, (segment_index, tajweed, _) in enumerate(segment_results):
        start = segment_index * segment_length_samples
        end = start + segment_length_samples
        if tajweed is not None:
            plt.plot(range(start, end), test_signal[start:end], color=colors(tajweeds.index(tajweed)), label=tajweed if tajweed not in legend_labels else "")
            legend_labels.add(tajweed)  # Tambahkan label ke set legenda
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Tajweed Detection in Test Audio')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()

def main():
    models_folder = 'models'
    test_audio_path = 'test.opus'
    test_features, test_signal, sr = extract_features(test_audio_path)
    if test_features is None or test_signal is None or sr is None:
        print("Failed to read test audio file.")
        return
    save_extracted_features(test_features, "test_features.txt")
    plot_audio_signal(test_signal, sr)
    segment_results, test_signal, sr, segment_length_samples = detect_tajweed(test_audio_path, models_folder)
    results_df = pd.DataFrame(segment_results, columns=['Segment', 'Tajweed', 'Similarity'])
    print(results_df)
    plot_detected_tajweed_segments(test_signal, segment_results, sr, segment_length_samples, [tajweed.split('_model')[0] for tajweed in os.listdir(models_folder) if tajweed.endswith('_model.pkl')])

if __name__ == "__main__":
    main()
