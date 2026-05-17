"""Fine-tune Ultralytics YOLO on ``dataset.yaml``; curves under ``model/<MODEL_NAME>/``."""
import argparse
import os
import sys
import tempfile

import yaml
from ultralytics import YOLO

from settings import DATASET_YAML, DEVICE, EPOCHS, MODEL_DIR, MODEL_NAME, YOLO_MODEL


def _resolve_data_paths(cfg_path, cfg):
    """Return absolute train/val image dirs from dataset yaml."""
    base = os.path.dirname(os.path.abspath(cfg_path))
    root = cfg.get("path", ".")
    root_abs = root if os.path.isabs(root) else os.path.normpath(os.path.join(base, root))
    train_rel = cfg.get("train")
    val_rel = cfg.get("val")
    if not train_rel or not val_rel:
        raise ValueError("dataset yaml must define 'train' and 'val'")
    train_abs = train_rel if os.path.isabs(train_rel) else os.path.normpath(os.path.join(root_abs, train_rel))
    val_abs = val_rel if os.path.isabs(val_rel) else os.path.normpath(os.path.join(root_abs, val_rel))
    return root_abs, train_abs, val_abs


def _write_ultralytics_data_yaml(cfg_path, cfg):
    """Ultralytics resolves ``path: .`` against cwd, not the yaml dir; write temp yaml with absolute ``path``."""
    root_abs, _, _ = _resolve_data_paths(cfg_path, cfg)
    merged = dict(cfg)
    merged["path"] = root_abs
    fd, name = tempfile.mkstemp(prefix="lab2_data_", suffix=".yaml", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
    return name


def main():
    p = argparse.ArgumentParser(description="Train YOLOv8 on dataset.yaml")
    p.add_argument("--epochs", type=int, default=EPOCHS, help=f"default {EPOCHS}")
    args = p.parse_args()
    if not os.path.isfile(DATASET_YAML):
        print(f"missing dataset yaml: {DATASET_YAML}", file=sys.stderr)
        sys.exit(1)

    with open(DATASET_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root_abs, train_abs, val_abs = _resolve_data_paths(DATASET_YAML, cfg)
    print(f"dataset root {root_abs}")
    print(f"train {train_abs}")
    print(f"val {val_abs}")
    if not os.path.isdir(train_abs):
        print(f"error: train images dir missing: {train_abs}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(val_abs):
        print(f"error: val images dir missing: {val_abs}", file=sys.stderr)
        sys.exit(1)

    run_name = MODEL_NAME or "train"
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Starting training script...")
    model = YOLO(YOLO_MODEL)
    print(f"Model loaded, training {args.epochs} epochs, device={DEVICE} ...")
    tmp_yaml = _write_ultralytics_data_yaml(DATASET_YAML, cfg)
    try:
        model.train(
            data=tmp_yaml,
            epochs=args.epochs,
            device=DEVICE,
            project=MODEL_DIR,
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
        )
    finally:
        try:
            os.remove(tmp_yaml)
        except OSError:
            pass
    run_dir = os.path.join(MODEL_DIR, run_name)
    print("Training finished")
    print(f"curves {os.path.join(run_dir, 'results.png')} {os.path.join(run_dir, 'results.csv')}")


if __name__ == "__main__":
    main()
