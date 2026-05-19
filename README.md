# Face Emotion Recognition with Live Webcam Overlay
Project Name: Face Emotion Recognition with Live Webcam Overlay
Roll No.: 23-AI-76
Dataset: FER-2013 — 7 emotions, ~35,000 grayscale 48×48 images
Course: Computer Vision Lab (Lab 14)

# Emotion Classes
angry · disgust · fear · happy · neutral · sad · surprise

# Setup Instructions
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn pandas

# Part 1 Train the Model (Google Colab)
The notebook will:

1. Download FER-2013 via kagglehub
2. Preprocess images (grayscale → Gaussian Blur → CLAHE → resize → normalize)
3. Build and train 3 architectures:
   CNN from Scratch (48×48 grayscale)
   MobileNetV2 Frozen (96×96 RGB, feature extractor)
   MobileNetV2 Fine-tuned (96×96 RGB, last 30 layers unfrozen)
   VGG-Style Double Conv Block (48×48 grayscale)
4. Evaluate each model with confusion matrix and per-class F1-score
5. Save the best model as model.keras

After training, download model.keras from Colab and place it in the project root.

# Part 2 Run Inference (VS Code / Local)
Make sure model.keras is in the same folder as main.py.

# 🎥 Live Webcam Mode
python main.py --webcam
Detects faces using Haar Cascade (haarcascade_frontalface_default.xml)
Classifies emotion per face with confidence bar overlay
Shows stable prediction using a rolling deque(maxlen=20)
Displays real-time FPS on screen
Press Q to quit and save a 30-second demo clip as demo_clip.avi

# Single Image Mode
python main.py

# OUTPUT
## Happy Test
<img width="365" height="329" alt="image" src="https://github.com/user-attachments/assets/c89badad-ac9e-41f2-bd10-ca277715660f" />

# 🏗️ Model Architectures
Model	Input	Params (approx)	Notes
CNN Scratch	48×48 grayscale	~1.2M	3 double-conv blocks + GAP
MobileNetV2 Frozen	96×96 RGB	~2.3M trainable	Backbone frozen, only head trained
MobileNetV2 Fine-tuned	96×96 RGB	~3.4M trainable	Last 30 backbone layers unfrozen
VGG-Style	48×48 grayscale	~1.8M	VGG-inspired double conv + BN

# Preprocessing Pipeline
Each face crop goes through:

BGR → Grayscale
Gaussian Blur (3×3 kernel)
CLAHE (clipLimit=2.0, tileGridSize=8×8)
Resize to 48×48 (or 96×96 for MobileNet)
Normalize to [0, 1]

# Evaluation Outputs
Per-class F1-score (via classification_report)
7×7 Confusion matrix (saved as PNG)
Model comparison table (accuracy)

# ⚠️ Known Limitations
Haar Cascade may miss faces at extreme angles (use DNN-based detector for better results)
disgust class is heavily underrepresented in FER-2013 (~550 samples vs 8000+ for happy)
Inference speed on CPU is ~5–10 FPS depending on hardware

# 📚 References
FER-2013 Dataset on Kaggle
MobileNetV2 Paper — Sandler et al., 2018
OpenCV Haar Cascade — haarcascade_frontalface_default.xml

