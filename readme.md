# Quran Tajweed Detection using DTW (Dynamic Time Warping)

## Introduction
**Tajweed** refers to the rules of proper pronunciation while reciting the Quran.  
**Dynamic Time Warping (DTW)** is an algorithm used to measure similarity between two sequences that may vary in speed or timing, making it suitable for audio comparison.

### Use Case
- Compare a student's recitation audio with a reference Quran recitation.  
- Detect pronunciation errors according to Tajweed rules.  
- Provide feedback for improvement.

---

## How DTW Works
1. Convert audio signals into **feature vectors** (e.g., MFCC – Mel-Frequency Cepstral Coefficients).  
2. Compute a distance matrix between the student’s audio and reference audio.  
3. Find the optimal alignment path that minimizes cumulative distance.  
4. Evaluate similarity: smaller distance = closer to correct Tajweed.

---

## Python Implementation (Simplified Example)
```python
import librosa
import numpy as np
from dtw import dtw

# Load reference and student audio
ref_audio, sr_ref = librosa.load("reference_quran.wav", sr=None)
student_audio, sr_stu = librosa.load("student_quran.wav", sr=None)

# Extract MFCC features
mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sr_ref, n_mfcc=13).T
mfcc_stu = librosa.feature.mfcc(y=student_audio, sr=sr_stu, n_mfcc=13).T

# Define Euclidean distance function
def euclidean(x, y):
    return np.linalg.norm(x - y)

# Compute DTW alignment
distance, cost_matrix, acc_cost_matrix, path = dtw(mfcc_ref, mfcc_stu, dist=euclidean)

print("DTW distance between reference and student recitation:", distance)

# Simple evaluation
threshold = 200  # example threshold, tune based on dataset
if distance < threshold:
    print("Recitation matches Tajweed rules.")
else:
    print("Pronunciation errors detected. Tajweed improvement needed.")
