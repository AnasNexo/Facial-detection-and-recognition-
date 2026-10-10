# FaceNet Real-Time Face Recognition

A complete end-to-end face recognition pipeline using **transfer learning** . Detects, preprocesses, trains, and recognises faces in real time from a webcam with name labels and confidence scores.

\---

##  How It Works

```
Webcam → MTCNN (detect) → FaceNet (128-d embedding) → SVM (classify) → Name + Confidence %
```

##  Project Structure

```
face-recognition/
│
├── face-dataset.py          # Step 1 — capture \~100 face images from webcam
├── face-preprocess.py       # Step 2 — align, crop, resize, normalise faces
├── face-train.py            # Step 3 — extract embeddings + train \& save SVM
├── face-recognition.py      # Step 4 — real-time recognition from webcam
├── requirements.txt
│
├── daniel/                  # raw face captures (auto-created)
├── daniel-preprocessed/     # 160×160 aligned crops (auto-created)
├── mark/
├── mark-preprocessed/
└── face\_model.pkl           # saved model (auto-created by face-train.py)
```

\---



### 1\. Install dependencies

```bash
pip install -r requirements.txt
```

> FaceNet weights (\~90 MB) are downloaded automatically on first run via `deepface`.

### 2\. Collect face images

```bash
python face-dataset.py --name daniel --samples 100
python face-dataset.py --name mark   --samples 100
```

* Look slightly left, right, up, down during capture for variety
* Good lighting makes a significant difference

### 3\. Preprocess

```bash
python face-preprocess.py --names daniel mark
# or auto-discover all folders:
python face-preprocess.py --auto
```

### 4\. Train

```bash
python face-train.py
```

Outputs `face\_model.pkl` with cross-validation accuracy printed to console.

### 5\. Recognise in real time

```bash
python face-recognition.py
```

Press **Q** to quit.

\---

##  Configuration

|Setting|File|Default|Notes|
|-|-|-|-|
|Confidence threshold|`face-recognition.py`|`0.70`|Faces below this show as Unknown|
|FaceNet variant|`face-train.py`|`"Facenet"` (128-d)|Change to `"Facenet512"` for higher accuracy|
|Samples to capture|`face-dataset.py`|`100`|More varied > more total|
|Frame skip|`face-recognition.py`|`1`|Set `--skip\_frames 2` on slow hardware|

\---

##  Performance Tips

* **More people = better Unknown rejection** — the SVM needs contrast between classes
* **Add an Unknown class** — collect 100 images of people who won't be recognised for the most reliable stranger rejection
* **Vary capture conditions** — different lighting, slight angle changes, glasses on/off
* Typical accuracy with 3–5 people and 100 images each: **85–95%**

\---

## Requirements

* Python 3.9+
* Webcam
* NVIDIA GPU optional (CPU works fine for inference)

```
opencv-python
mtcnn
deepface
tensorflow
scikit-learn
numpy
joblib
Pillow
```

\---

## Model Architecture

**FaceNet (Inception-ResNet-v1)**

* Input: 160×160 RGB face crop, normalised to zero mean / unit variance
* Output: 128-dimensional L2-normalised embedding vector
* Trained with Triplet Loss on VGGFace2 (3.3M images, 9,131 identities)
* Used here as a **frozen feature extractor** (transfer learning — no retraining)

**SVM Classifier**

* Pipeline: L2 normalise → RBF-kernel SVM with Platt scaling
* Trained on your embeddings only (\~100 images × number of people)
* `predict\_proba()` gives per-class confidence scores shown on screen

\---
