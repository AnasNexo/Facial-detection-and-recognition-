# Face recognition: PCA and FaceNet512

I built this project to compare two ways of recognizing faces: learning a compact representation with PCA, and using embeddings from a pretrained FaceNet512 model. Both approaches use the same images and the same train/test split, so I can compare the classifiers on a common basis.

The workflow covers image quality checks, MTCNN detection, face selection, eye alignment, training, and evaluation. The notebooks include the reasoning behind the preprocessing choices and a closer look at the mistakes each model makes.

## Results

The experiment starts with 500 LFW images across 10 identities. After one manual exclusion and one low-sharpness rejection, 498 images remain: 398 for training and 100 for testing, with 10 test images per identity.

| Approach | Test accuracy | Macro precision | Macro recall | Macro F1 | CV macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FaceNet512 + linear SVM | 95% | 95.18% | 95% | 94.99% | 92.96% |
| FaceNet512 + RBF SVM | 95% | 95.18% | 95% | 94.99% | 93.71% |
| FaceNet512 + kNN | 93% | 93.59% | 93% | 93.02% | 91.08% |
| PCA + linear SVM | 96% | 96.42% | 96% | 95.98% | 84.26% |
| PCA + RBF SVM | 97% | 97.27% | 97% | 96.93% | 87.99% |

![Test metrics for the five models](analyse/figures/test_metrics.png)

PCA with an RBF SVM has the highest accuracy on this particular test split. FaceNet with an RBF SVM has the strongest cross-validation macro F1. With only 100 test images, one mistake changes accuracy by one percentage point. The paired comparisons do not establish a statistically significant winner after Holm correction.

These are results for classification among ten known identities. They do not measure recognition of unknown people, face verification, or live camera performance.

## Project layout

```text
analyse/
  load_lfw.py                       Download and select the LFW sample
  prepare_data.py                   Recreate the pre-filter working dataset
  preprocess_analyse_ameliore.ipynb Quality checks, selection galleries and shared split
  preprocess_pca.py                Aligned 80 x 80 grayscale inputs
  preprocess_facenet.py            Aligned 160 x 160 RGB inputs
  train_pca.py                     PCA with linear and RBF SVM
  train_facenet_transfert.py       Frozen FaceNet512 with SVM and kNN
  analyse_resultat.ipynb           Executed English results notebook
  analyse_resultat.py              Plot and table helpers for the results notebook
  reference_split/                 Original split, source hashes and duplicate candidates
  results/                         Numeric snapshot of the completed experiment
  figures/                         English comparison plots
  REPORT.md                        Evaluation protocol and interpretation
```

Datasets, pretrained weights, fitted models, and embedding caches are excluded from Git. The results notebook has saved outputs, so its tables and charts are visible on GitHub without downloading the images. The preprocessing notebook is included without its large image-gallery outputs; run it locally to inspect the sources and selected faces.

## Run the project

Use Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Then install the dependencies and prepare the sample:

```bash
python -m pip install -r requirements.txt
python analyse/load_lfw.py
python analyse/prepare_data.py
python -m notebook
```

Open `analyse/preprocess_analyse_ameliore.ipynb` and run it from top to bottom. Inspect the ambiguous and multiple-face galleries. Section 11 backs up and removes sources with Laplacian variance below 20, then exports one face per source. Section 13 creates the shared stratified 80/20 split with seed 42 and checks for duplicate leakage.

The preparation script reproduces the recorded manual exclusion, `George_W_Bush_0039.jpg`. The sharpness filter rejects `John_Ashcroft_0043.jpg` in the recorded run. The committed reference manifest records the exact original membership and pixel hashes; compare regenerated manifests with it before interpreting a rerun as the same experiment. Detector and numerical-library changes can affect results.

Prepare the two input formats and train:

```bash
python analyse/preprocess_pca.py
python analyse/preprocess_facenet.py
python -u analyse/train_pca.py
python -u analyse/train_facenet_transfert.py
```

Training progress prints in the terminal and is saved in each model's results directory. The first FaceNet run downloads its weights. CPU execution works; embedding extraction takes most of the FaceNet runtime.

PCA results are written to `analyse/preprocessed_crop_pca_split/pca_results/`. FaceNet results go to `analyse/cropped_faces_facenet_split/facenet_transfer_results/`. Each contains metrics, predictions, confusion matrices, settings, and fitted classifiers. The committed `analyse/results/` directory is a snapshot of the completed experiment; a new training run writes to the local directories above and does not silently replace that snapshot.

To view or regenerate the published comparison charts, run `analyse/analyse_resultat.ipynb`. It reads the included numeric snapshot and requires no face images or model weights. See [the full report](analyse/REPORT.md) for validation details and limitations.

## Preprocessing and training choices

- **Face selection:** MTCNN detects faces and landmarks. A heuristic combines relative box area (60%), centrality (35%), and confidence (5%). One face is retained per source, and ambiguous cases can be reviewed in a gallery.
- **PCA:** align eyes from the original source, resize to 80 x 80, convert to grayscale, and scale pixels to `[0, 1]`. PCA is fitted inside each cross-validation fold. No CLAHE or gamma correction is applied.
- **FaceNet512:** align the same selected face from source landmarks and save a 160 x 160 RGB PNG with pixels in `[0, 255]`. DeepFace applies `normalization="Facenet"` during `represent`, with `detector_backend="skip"` and `align=False`. L2-normalize the 512-dimensional embeddings before classification.
- **Transfer learning:** FaceNet weights remain frozen. Only the SVM or kNN classifier is fitted; no FaceNet fine-tuning is performed.
- **Validation:** select hyperparameters by macro F1 using five training folds, grouping flagged near-duplicate sources together. The two input families retain the same train/test membership.

## What I would test next

I would collect a larger, independent test set before changing the pipeline based on these results. Then I would compare stronger regularization for PCA, inspect the recurring errors and face selections, and test more identities and different capture conditions. Fine-tuning FaceNet would need substantially more varied training data and a separate validation protocol.

## Data and dependencies

The images come from [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/). `load_lfw.py` downloads the funneled archive from the mirror used by scikit-learn and verifies its SHA-256 checksum. Images are not redistributed in this repository; consult the dataset's terms before using them.

This project uses [DeepFace](https://github.com/serengil/deepface), [MTCNN](https://github.com/ipazc/mtcnn), [scikit-learn](https://scikit-learn.org/), OpenCV, TensorFlow, NumPy, pandas, and Matplotlib. Their code and pretrained weights remain subject to their respective licenses and terms.

## Previous version

The earlier webcam-based project is preserved on [`archive/webcam-face-recognition`](https://github.com/AnasNexo/Facial-detection-and-recognition-/tree/archive/webcam-face-recognition).
