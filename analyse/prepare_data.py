"""Copy the downloaded sample into the working dataset before notebook cleaning.

Run after load_lfw.py, before the preprocessing notebook. The single manual
exclusion records the original experiment; raw downloaded images stay intact.
"""
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
MANUAL_EXCLUSIONS = {"George_W_Bush/George_W_Bush_0039.jpg"}


def main():
    source = BASE / "data"
    target = BASE / "preprocessed_data"
    images = sorted(source.glob("*/*.jpg"))
    if len(images) != 500:
        raise ValueError("Expected the 500-image sample; run analyse/load_lfw.py first")
    count = 0
    for image in images:
        relative = image.relative_to(source)
        if relative.as_posix() in MANUAL_EXCLUSIONS:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != image.read_bytes():
            raise FileExistsError(f"Working image differs from raw source: {relative}")
        shutil.copy2(image, destination)
        count += 1
    print(f"Prepared {count} images in {target}")
    print("Next: run preprocess_analyse_ameliore.ipynb, including filtering and the split.")
    print("Rerunning this command restores the pre-filter baseline from the raw sample.")


if __name__ == "__main__":
    main()
