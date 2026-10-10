import argparse
import csv
import hashlib
import io
from collections import defaultdict
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request

from PIL import Image


SOURCE = "https://ndownloader.figshare.com/files/5976015"
SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"


def export(archive_path, output_dir):
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("LFW archive SHA-256 does not match the published checksum")

    with tarfile.open(archive_path, "r:gz") as archive:
        identities = defaultdict(list)
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if (member.isfile() and len(parts) == 3
                    and parts[0] == "lfw_funneled"
                    and parts[1] not in (".", "..")
                    and parts[2].lower().endswith(".jpg")):
                identities[parts[1]].append(member)

        # Choose well-represented identities, with deterministic tie-breaking.
        selected = sorted(
            (name for name, files in identities.items() if len(files) >= 50),
            key=lambda name: (-len(identities[name]), name),
        )[:10]
        if len(selected) != 10:
            raise ValueError("Archive has fewer than 10 identities with 50 images")

        rows = []
        for label_id, name in enumerate(sorted(selected)):
            members = sorted(identities[name], key=lambda member: member.name)[:50]
            for member in members:
                relative_path = PurePosixPath(name) / PurePosixPath(member.name).name
                payload = archive.extractfile(member).read()
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                destination = output_dir.joinpath(*relative_path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.read_bytes() != payload:
                    raise FileExistsError(f"Refusing to replace existing image: {destination}")
                destination.write_bytes(payload)
                rows.append({
                    "path": relative_path.as_posix(),
                    "label_id": label_id,
                    "celebrity": name.replace("_", " "),
                })
            print(f"{name.replace('_', ' ')}: 50 images", flush=True)

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["path", "label_id", "celebrity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} labeled images to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing LFW funneled .tgz archive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.archive:
        export(args.archive, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="lfw-") as temporary:
            archive_path = Path(temporary) / "lfw-funneled.tgz"
            print(f"Downloading {SOURCE}", flush=True)
            with urllib.request.urlopen(SOURCE, timeout=120) as response:
                with archive_path.open("wb") as target:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        target.write(chunk)
            export(archive_path, args.output_dir)


if __name__ == "__main__":
    main()
