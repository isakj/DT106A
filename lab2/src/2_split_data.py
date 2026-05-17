"""Shuffle and split YOLO-format images/labels into train/valid/test folders.

Resolves Roboflow-style label filenames ({stem}_{ext}.rf.*.txt), converts polygon
rows to axis-aligned bounding boxes, and optionally filters by class id. Skips
images with no label file and images with nothing left after filtering.
"""
import argparse
import glob
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from settings import CWDIR, SPLIT_RATIOS

DATASET_DIR = os.path.normpath(os.path.join(CWDIR, "..", "dataset"))
_ORG = os.path.join(DATASET_DIR, "org_data")
if os.path.isdir(os.path.join(_ORG, "images")):
    IMAGES_DIR = os.path.join(_ORG, "images")
    LABELS_DIR = os.path.join(_ORG, "labels")
else:
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    LABELS_DIR = os.path.join(DATASET_DIR, "labels")


def check_directories():
    """Return True if IMAGES_DIR and LABELS_DIR exist; otherwise print and return False."""
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(LABELS_DIR):
        print("Images or labels directory not found. Expected under dataset or dataset/org_data.")
        return False
    return True


def delete_existing_splits():
    """Remove prior split directories under DATASET_DIR (keys from SPLIT_RATIOS)."""
    for split in SPLIT_RATIOS:
        split_dir = os.path.join(DATASET_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            print(f"Deleted existing {split} split.")


def create_dirs():
    """Create empty images/labels subfolders for each split name in SPLIT_RATIOS."""
    for split in SPLIT_RATIOS:
        os.makedirs(os.path.join(DATASET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, "labels"), exist_ok=True)


def find_label_path(img_name):
    """Return absolute path to the label file for an image basename, or None.

    Tries Roboflow pattern ``{stem}_{ext}.rf.*.txt`` then plain ``{stem}.txt``.
    """
    stem, ext = os.path.splitext(img_name)
    ext_token = ext.lstrip(".").lower() or "jpg"
    pat = os.path.join(LABELS_DIR, f"{stem}_{ext_token}.rf.*.txt")
    m = glob.glob(pat)
    if m:
        return m[0]
    plain = os.path.join(LABELS_DIR, f"{stem}.txt")
    if os.path.isfile(plain):
        return plain
    return None


def line_to_bbox(parts):
    """Parse one YOLO label line (split tokens) into (cls, cx, cy, w, h) normalized.

    ``parts[0]`` is class id; remaining tokens are either four bbox values or an
    even-length polygon x,y,... sequence. Returns None if the row is invalid.
    """
    cls = int(parts[0])
    nums = [float(x) for x in parts[1:]]
    n = len(nums)
    if n == 4:
        return cls, nums[0], nums[1], nums[2], nums[3]
    if n >= 6 and n % 2 == 0:
        xs = nums[0::2]
        ys = nums[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cls, cx, cy, w, h
    return None


def filter_and_bbox_label(text, class_filter):
    """Filter label text by class id set and rewrite every line as YOLO bbox format.

    Args:
        text: Full contents of a label file.
        class_filter: Set of class ids to keep, or None to keep all.

    Returns:
        New file body (possibly empty) with one ``cls cx cy w h`` per line.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            continue
        if class_filter is not None and cls not in class_filter:
            continue
        bb = line_to_bbox(parts)
        if bb is None:
            continue
        cls, cx, cy, w, h = bb
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def dominant_class_for_stratum(filtered):
    """Class id with the most boxes in this label (tie → smallest id)."""
    c = Counter()
    for line in filtered.splitlines():
        line = line.strip()
        if not line:
            continue
        c[int(line.split()[0])] += 1
    if not c:
        return 0
    mx = max(c.values())
    return min(cl for cl, v in c.items() if v == mx)


def stratified_splits(records):
    """Assign records to train/valid/test with same ratio per dominant-class stratum."""
    by_dc = defaultdict(list)
    for img, filtered in records:
        by_dc[dominant_class_for_stratum(filtered)].append((img, filtered))
    train, valid, test = [], [], []
    r_tr = SPLIT_RATIOS["train"]
    r_va = SPLIT_RATIOS["valid"]
    for _, lst in sorted(by_dc.items()):
        random.shuffle(lst)
        n = len(lst)
        te = int(n * r_tr)
        ve = te + int(n * r_va)
        train.extend(lst[:te])
        valid.extend(lst[te:ve])
        test.extend(lst[ve:])
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return {"train": train, "valid": valid, "test": test}


def collect_usable_records(class_filter):
    """Pairs (image_name, filtered_label_text) for images with a label and ≥1 box after filter."""
    recs = []
    for img in os.listdir(IMAGES_DIR):
        if not img.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lp = find_label_path(img)
        if not lp:
            continue
        with open(lp, encoding="utf-8") as f:
            body = f.read()
        filtered = filter_and_bbox_label(body, class_filter)
        if not filtered.strip():
            continue
        recs.append((img, filtered))
    return recs


def split_data(class_filter):
    """Copy images into split folders and write filtered bbox labels next to them.

    Only images with a resolvable label file and at least one annotation after
    class filtering are included. Train/valid/test are stratified by dominant
    class per image so each split gets a similar mix of classes.

    Args:
        class_filter: Set of YOLO class ids to retain, or None for all classes.
    """
    records = collect_usable_records(class_filter)
    n = len(records)
    splits = stratified_splits(records)

    def out_label_name(img):
        return os.path.splitext(img)[0] + ".txt"

    for split, items in splits.items():
        for img, filtered in items:
            shutil.copy(
                os.path.join(IMAGES_DIR, img),
                os.path.join(DATASET_DIR, split, "images", img),
            )
            dst_l = os.path.join(DATASET_DIR, split, "labels", out_label_name(img))
            with open(dst_l, "w", encoding="utf-8") as f:
                f.write(filtered)

    print(f"Dataset split complete! ({n} images with labels after class filter)")


def count_files():
    """Print per-split image count and box counts per YOLO class."""
    for split in SPLIT_RATIOS:
        idir = os.path.join(DATASET_DIR, split, "images")
        ldir = os.path.join(DATASET_DIR, split, "labels")
        img_count = len(os.listdir(idir)) if os.path.isdir(idir) else 0
        c = Counter()
        if os.path.isdir(ldir):
            for fn in os.listdir(ldir):
                if not fn.endswith(".txt"):
                    continue
                with open(os.path.join(ldir, fn), encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            c[int(line.split()[0])] += 1
                        except (ValueError, IndexError):
                            pass
        if c:
            inner = ", ".join(f"class {k}: {v}" for k, v in sorted(c.items()))
            extra = f" ({inner})"
        else:
            extra = " (no boxes)" if img_count else " ()"
        print(f"{split.capitalize()} - {img_count} images,{extra}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Split dataset into train/valid/test.")
    p.add_argument(
        "classes",
        nargs="*",
        type=int,
        metavar="CLASS",
        help="Optional YOLO class ids to keep (one integer per argument). Omit for all.",
    )
    args = p.parse_args()
    cf = set(args.classes) if args.classes else None

    if not check_directories():
        sys.exit(1)
    delete_existing_splits()
    create_dirs()
    split_data(cf)
    count_files()
