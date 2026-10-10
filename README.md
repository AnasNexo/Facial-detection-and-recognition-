# Face recognition with FaceNet

This project collects face images from a webcam and trains a classifier to recognise the people in those images. MTCNN detects faces, FaceNet produces a 128-value embedding for each face, and an SVM predicts the person's name.

FaceNet uses pretrained weights. Training fits the SVM to the people in your dataset. During webcam recognition, the program displays a name and a classifier confidence score.

## Setup

Use Python 3.10 or newer and a webcam. Run the following from the project folder:

```bash
python -m pip install -r requirements.txt tf-keras
```

DeepFace downloads the FaceNet weights the first time they are needed, so the first training run requires an internet connection.

## Collect images

Create a folder of webcam captures for each person. For example:

```bash
python face-dataset.py --name daniel --samples 100
python face-dataset.py --name mark --samples 100
```

The images are saved in `daniel/` and `mark/`. Move your head slightly while collecting samples and try different lighting and expressions. Press `q` to stop capture early.

## Prepare the faces

Open `face-preprocess.py` and set the `names` list near the bottom to match your folders:

```python
names = ["daniel", "mark"]
```

Then run:

```bash
python face-preprocess.py
```

The script detects and aligns faces, then saves 160 by 160 pixel crops in folders such as `daniel-preprocessed/` and `mark-preprocessed/`. Check these images before training, especially any failed detections or poor crops reported by the script.

## Train the classifier

You need images for at least two people. Run:

```bash
python face-train.py
```

The script reads the `*-preprocessed` folders, extracts FaceNet embeddings, and trains an SVM. It prints cross-validation accuracy when enough samples are available and saves the classifier and labels in `face_model.pkl`.

## Run webcam recognition

```bash
python face-recognition.py
```

Press `q` to close the webcam window. To use another camera or reduce how often detection runs:

```bash
python face-recognition.py --camera 1 --skip_frames 2
```

You can load a different saved model with `--model path/to/model.pkl`.

The current recognition threshold is `0.70` in `face-recognition.py`. Predictions below it are displayed as `Unknown`. This score comes from the classifier; it should be checked against separate test images before choosing a threshold for your dataset.

## Files

| File | Purpose |
| --- | --- |
| `face-dataset.py` | Collect webcam images for a person |
| `face-preprocess.py` | Detect, align, and resize faces |
| `face-train.py` | Extract embeddings and train the SVM |
| `face-recognition.py` | Recognise faces from a webcam |
| `requirements.txt` | Python dependencies |
| `index.html` | Project overview page |

Image folders, downloaded model weights, and the saved classifier are kept out of Git.

## Dataset analysis

The [analysis branch](https://github.com/AnasNexo/Facial-detection-and-recognition-/tree/analysis/image-quality-eda) contains the LFW loader and the [EDA notebook](https://github.com/AnasNexo/Facial-detection-and-recognition-/blob/analysis/image-quality-eda/analyse/preprocess.ipynb).

The notebook covers class counts, image dimensions, exact and near duplicates, lighting, contrast, sharpness, and MTCNN detection measurements. Its tables and example images are saved inline. Downloaded images and generated report folders remain local.

To run the notebook on that branch, install its additional dependencies and select the same Python environment in your notebook editor:

```bash
python -m pip install pandas matplotlib ipykernel
```

## Working with the dataset

Keep images from separate capture sessions for evaluation. Photos taken a fraction of a second apart can be almost identical, which can make random train/test splits look better than performance on new images.

Review duplicate images and failed face detections before training. Accuracy depends on the people, lighting, image quality, and evaluation split; the project does not include a verified accuracy benchmark.
