import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    scaler = StandardScaler()
    mfccs = scaler.fit_transform(mfccs.T).T  # Normalisasi fitur
    return mfccs

def save_model(features, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Model saved to {file_path}")

def main():
    data_latih_folder = 'data_latih/tajweed'
    models_folder = 'models'
    os.makedirs(models_folder, exist_ok=True)
    
    for tajweed_folder in os.listdir(data_latih_folder):
        tajweed_path = os.path.join(data_latih_folder, tajweed_folder)
        if not os.path.isdir(tajweed_path):
            continue
        
        tajweed_features = []
        for file_name in os.listdir(tajweed_path):
            file_path = os.path.join(tajweed_path, file_name)
            if file_path.endswith('.opus'):
                features = extract_features(file_path)
                tajweed_features.append(features)
        
        # Ensure all features have the same shape by padding or trimming
        max_length = max(feature.shape[1] for feature in tajweed_features)
        tajweed_features = [np.pad(feature, ((0, 0), (0, max_length - feature.shape[1])), 'constant') for feature in tajweed_features]
        
        model_path = os.path.join(models_folder, f'{tajweed_folder}_model.pkl')
        save_model(tajweed_features, model_path)

if __name__ == "__main__":
    main()
