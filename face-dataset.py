"""
face-dataset.py
---------------
Opens the webcam, detects a face using MTCNN, and captures ~100 images
of that face. Saves them in a folder named after the person.

Usage:
    python face-dataset.py --name daniel --samples 100

Directory output:
    daniel/
        0001.jpg
        0002.jpg
        ...
        0100.jpg

Requirements:
    pip install opencv-python mtcnn tensorflow numpy
"""

import cv2
import os
import argparse
import time
import numpy as np
from mtcnn import MTCNN


def collect_face_dataset(name: str, samples: int = 100, output_dir: str = "."):
    """
    Capture face images from webcam using MTCNN for detection.

    Args:
        name: Person's name (used as folder name)
        samples: Number of images to capture (default 100)
        output_dir: Root directory to save images in
    """
    # Create output directory: ./daniel/
    save_dir = os.path.join(output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check your camera connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[INFO] Starting capture for: '{name}'")
    print(f"[INFO] Target: {samples} face images → saved to '{save_dir}/'")
    print("[INFO] Press 'q' to quit early.\n")

    count = 0
    last_capture_time = 0
    capture_interval = 0.15  # seconds between captures (avoid duplicate frames)

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        display_frame = frame.copy()
        current_time = time.time()

        # Detect faces (MTCNN expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)

        face_detected = False

        if results:
            # Use the face with the highest confidence
            best = max(results, key=lambda x: x["confidence"])

            if best["confidence"] > 0.92:
                face_detected = True
                x, y, w, h = best["box"]

                # Clamp coordinates to frame boundaries
                x, y = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

                # Add padding around the face (20%)
                pad_w = int(w * 0.2)
                pad_h = int(h * 0.2)
                x_pad = max(0, x - pad_w)
                y_pad = max(0, y - pad_h)
                x2_pad = min(frame.shape[1], x2 + pad_w)
                y2_pad = min(frame.shape[0], y2 + pad_h)

                face_crop = frame[y_pad:y2_pad, x_pad:x2_pad]

                # Save at interval to ensure image variety
                if current_time - last_capture_time >= capture_interval and face_crop.size > 0:
                    filename = os.path.join(save_dir, f"{count + 1:04d}.jpg")
                    cv2.imwrite(filename, face_crop)
                    count += 1
                    last_capture_time = current_time
                    print(f"  Captured {count}/{samples}: {filename}")

                # Draw bounding box and landmarks on display
                cv2.rectangle(display_frame, (x, y), (x2, y2), (0, 255, 0), 2)
                for key, point in best["keypoints"].items():
                    cv2.circle(display_frame, point, 3, (0, 200, 255), -1)

                conf_text = f"Face: {best['confidence']:.2f}"
                cv2.putText(display_frame, conf_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # HUD overlay
        progress = int((count / samples) * 200)
        cv2.rectangle(display_frame, (10, 10), (10 + 200, 30), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (10, 10), (10 + progress, 30), (0, 220, 80), -1)
        cv2.putText(display_frame, f"{count}/{samples} captured", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        status_text = "FACE DETECTED" if face_detected else "NO FACE"
        cv2.putText(display_frame, status_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.putText(display_frame, f"Subject: {name}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

        cv2.imshow("Face Dataset Collection — press Q to quit", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Early exit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Captured {count} images for '{name}' → '{save_dir}/'")
    if count < samples:
        print(f"[WARN] Only captured {count}/{samples}. Re-run to collect more.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect face dataset images from webcam using MTCNN."
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Person's name (used as subfolder name, e.g. 'daniel')"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of face images to capture (default: 100)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Root directory for dataset (default: current directory)"
    )
    args = parser.parse_args()

    collect_face_dataset(
        name=args.name,
        samples=args.samples,
        output_dir=args.output_dir
    )
