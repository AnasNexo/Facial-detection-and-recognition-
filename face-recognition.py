"""
face-recognition.py
--------------------
Real-time face recognition from webcam using:
  • MTCNN  — face detection & landmark extraction
  • FaceNet — 128-d embedding (transfer learning, frozen weights)
  • SVM    — classification with probability (confidence) scores

Displays bounding box + person name + confidence % on screen.
Faces below the unknown_threshold are labelled "Unknown".

Usage:
    python face-recognition.py
    python face-recognition.py --model face_model.pkl --camera 0

Requirements:
    pip install deepface tensorflow scikit-learn opencv-python mtcnn numpy joblib
"""

import cv2
import numpy as np
import joblib
import argparse
import time
from mtcnn import MTCNN
from deepface import DeepFace


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_embedding_from_crop(face_bgr: np.ndarray, model_name: str) -> np.ndarray | None:
    """
    Extract FaceNet embedding from a BGR face crop.

    Returns:
        numpy array of shape (128,) or None on failure
    """
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path=face_rgb,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",
        )
        return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception:
        return None


def draw_label(
    frame: np.ndarray,
    label: str,
    confidence: float,
    x1: int, y1: int, x2: int, y2: int,
    is_known: bool
) -> None:
    """Draw bounding box and name/confidence label on the frame."""
    color = (0, 220, 80) if is_known else (0, 60, 220)   # green = known, red = unknown
    thickness = 2

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    text = f"{label}  {confidence * 100:.1f}%"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    bg_y1 = max(y1 - th - baseline - 8, 0)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - baseline - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Facial landmarks (small dots)
    pass  # landmarks drawn separately in main loop


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_recognition(model_path: str, camera_id: int, skip_frames: int = 1):
    """
    Real-time face recognition loop.

    Args:
        model_path: Path to the saved model bundle (.pkl)
        camera_id: OpenCV camera index (usually 0)
        skip_frames: Process every N-th frame (1 = every frame)
                     Increase to 2-3 on slow hardware.
    """
    print(f"[INFO] Loading model from '{model_path}'...")
    bundle = joblib.load(model_path)
    clf           = bundle["classifier"]
    le            = bundle["label_encoder"]
    threshold     = 0.70
    facenet_model = bundle["facenet_model"]
    people        = bundle["people"]

    print(f"[INFO] Recognising: {people}")
    print(f"[INFO] Unknown threshold: {threshold}")
    print(f"[INFO] FaceNet model: {facenet_model}\n")

    print("[INFO] Initialising MTCNN detector...")
    detector = MTCNN()

    print("[INFO] Warming up FaceNet (first call downloads weights if needed)...")
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    extract_embedding_from_crop(dummy, facenet_model)
    print("[INFO] Ready.\n")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    print("[INFO] Starting real-time recognition — press Q to quit.\n")

    frame_count   = 0
    last_results  = []   # cached detections for skipped frames
    fps_timer     = time.time()
    fps           = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            break

        frame_count += 1

        # ── FPS calculation ───────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        # ── Detection + recognition (every `skip_frames` frames) ─────────
        if frame_count % skip_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb)
            last_results = []

            for det in detections:
                if det["confidence"] < 0.90:
                    continue

                x, y, w, h = det["box"]
                x, y = max(0, x), max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                if (x2 - x) < 40 or (y2 - y) < 40:
                    continue  # too small to be reliable

                # Add 15% padding for better embedding quality
                pad = int(min(w, h) * 0.15)
                xp  = max(0, x - pad)
                yp  = max(0, y - pad)
                xp2 = min(frame.shape[1], x2 + pad)
                yp2 = min(frame.shape[0], y2 + pad)

                face_crop = frame[yp:yp2, xp:xp2]
                embedding = extract_embedding_from_crop(face_crop, facenet_model)

                if embedding is None:
                    continue

                # Predict
                emb_2d = embedding.reshape(1, -1)
                proba  = clf.predict_proba(emb_2d)[0]
                top_idx = int(np.argmax(proba))
                confidence = float(proba[top_idx])
                name = le.inverse_transform([top_idx])[0]

                is_known = confidence >= threshold
                display_name = name if is_known else "Unknown"

                last_results.append({
                    "name": display_name,
                    "confidence": confidence,
                    "is_known": is_known,
                    "box": (x, y, x2, y2),
                    "keypoints": det["keypoints"],
                    "all_proba": {le.classes_[i]: float(proba[i]) for i in range(len(proba))},
                })

        # ── Render cached results ──────────────────────────────────────────
        for res in last_results:
            x1, y1, x2, y2 = res["box"]
            draw_label(frame, res["name"], res["confidence"],
                       x1, y1, x2, y2, res["is_known"])

            # Facial landmarks
            for pt_name, pt in res["keypoints"].items():
                cv2.circle(frame, pt, 3, (0, 200, 255), -1)

            # Detailed probability bar (top-right of bounding box)
            bar_x = x2 + 10
            if bar_x + 150 < frame.shape[1]:
                for i, (pname, pval) in enumerate(
                    sorted(res["all_proba"].items(), key=lambda kv: -kv[1])
                ):
                    bar_y = y1 + i * 22
                    bar_len = int(pval * 120)
                    bar_color = (0, 200, 80) if pname == res["name"] else (80, 80, 80)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 16),
                                  bar_color, -1)
                    cv2.putText(frame, f"{pname[:10]} {pval*100:.1f}%",
                                (bar_x, bar_y + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        draw_fps(frame, fps)

        # Controls hint
        cv2.putText(frame, "Q: quit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Face Recognition — FaceNet + SVM", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time face recognition using FaceNet + SVM."
    )
    parser.add_argument(
        "--model", type=str, default="face_model.pkl",
        help="Path to trained model bundle (default: face_model.pkl)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--skip_frames", type=int, default=1,
        help="Run detection every N frames (default: 1 = every frame). "
             "Increase to 2-3 on slow hardware."
    )
    args = parser.parse_args()

    run_recognition(
        model_path=args.model,
        camera_id=args.camera,
        skip_frames=args.skip_frames,
    )
