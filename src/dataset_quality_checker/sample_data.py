"""Generate a small synthetic YOLO dataset (no downloads needed).

``with_defects=False`` produces a clean dataset that should pass every check.
``with_defects=True`` additionally plants one example of each defect type so
you can see every validator fire. All content is deterministic for a seed.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

NUM_CLASSES = 3
_COLORS = [(220, 50, 50), (50, 180, 70), (60, 90, 220)]
YoloBox = Tuple[int, float, float, float, float]


def _scene(rng: random.Random, classes: List[int]) -> Tuple[Image.Image, List[YoloBox]]:
    """Draw one coloured rectangle per requested class on a random background."""
    width, height = rng.randint(160, 320), rng.randint(160, 320)
    bg = tuple(rng.randint(150, 255) for _ in range(3))
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    boxes: List[YoloBox] = []
    for cls in classes:
        bw, bh = rng.randint(30, width // 2), rng.randint(30, height // 2)
        x0, y0 = rng.randint(0, width - bw), rng.randint(0, height - bh)
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=_COLORS[cls % NUM_CLASSES])
        boxes.append((cls, (x0 + bw / 2) / width, (y0 + bh / 2) / height, bw / width, bh / height))
    return img, boxes


def _fmt(boxes: List[YoloBox]) -> str:
    return "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes) + "\n"


def generate_sample_dataset(
    out_dir: Path, *, with_defects: bool = True, num_images: int = 24, seed: int = 42
) -> Path:
    """Create the dataset in ``out_dir`` (must not exist or be empty).

    Raises:
        FileExistsError: ``out_dir`` exists and is not empty (we never overwrite).
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty folder: {out_dir}")
    rng = random.Random(seed)

    def put(split: str, stem: str, img: Image.Image, label_text: str | None) -> None:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        img.save(out_dir / "images" / split / f"{stem}.png")
        if label_text is not None:
            (out_dir / "labels" / split / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    # --- healthy images ------------------------------------------------
    for i in range(num_images):
        split = "train" if i % 4 else "val"
        if with_defects:  # deliberately imbalanced: class 2 is rare
            if i % 2:
                classes = [0, 0]
            elif i == 6:
                classes = [0, 2]
            else:
                classes = [0] + ([1] if i % 4 == 0 else [])
        else:
            classes = [i % 3, (i + 1) % 3]
        img, boxes = _scene(rng, classes)
        put(split, f"img_{i:04d}", img, _fmt(boxes))

    if not with_defects:
        return out_dir

    # --- planted defects -----------------------------------------------
    good = "0 0.5 0.5 0.3 0.3\n"
    train = out_dir / "images" / "train"

    (train / "corrupt_not_an_image.jpg").write_bytes(b"this is definitely not a jpeg")
    (out_dir / "labels" / "train" / "corrupt_not_an_image.txt").write_text(good)

    full = train / "img_0001.png"
    data = full.read_bytes()
    (train / "truncated.png").write_bytes(data[: len(data) // 2])
    (out_dir / "labels" / "train" / "truncated.txt").write_text(good)

    img, _ = _scene(rng, [0])
    put("train", "no_label", img, None)

    put("train", "tiny", Image.new("RGB", (8, 8), (10, 200, 10)), good)
    put("train", "blank", Image.new("RGB", (128, 128), (255, 255, 255)), "")

    shutil.copy(train / "img_0001.png", train / "img_0001_copy.png")
    shutil.copy(out_dir / "labels" / "train" / "img_0001.txt", out_dir / "labels" / "train" / "img_0001_copy.txt")

    bad_labels = {
        "bad_negative_class": "-1 0.5 0.5 0.2 0.2\n",
        "bad_class_out_of_range": "7 0.5 0.5 0.2 0.2\n",
        "bad_class_not_integer": "1.5 0.5 0.5 0.2 0.2\n",
        "bad_zero_width": "0 0.5 0.5 0.0 0.2\n",
        "bad_out_of_bounds": "0 0.9 0.9 0.4 0.4\n",
        "bad_pixel_coordinates": "0 120 80 60 40\n",
        "bad_malformed_line": "0 0.5 0.5\n",
        "bad_not_numeric": "0 abc 0.5 0.2 0.2\n",
        "bad_duplicate_box": "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
    }
    for stem, text in bad_labels.items():
        img, _ = _scene(rng, [0])
        put("train", stem, img, text)

    (out_dir / "labels" / "train" / "ghost_orphan.txt").write_text(good)
    return out_dir
