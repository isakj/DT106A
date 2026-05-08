# Student Boilerplate: Visual AI Lab 2

This is your starting point for Lab 2.

In this lab you will build a complete object detection pipeline using YOLOv8:

1. **Capture** — record training images with your webcam
2. **Split** — divide images into train / validation / test sets
3. **Train** — train/fine-tune a YOLOv8 model on your images
4. **Infer** — run live detection with your trained model

Follow the numbered scripts in order. Look for `TODO` markers and complete informations in the all python files under `src/*.py`.

---

## Folder Overview
|------------|---------------------------------------------------------|
| Folder     | Purpose                                                 |
|------------|---------------------------------------------------------|
| `src/`     | Numbered pipeline scripts — work through these in order |
| `dataset/` | Input images, Input labels, and train/valid/test splits |
| `model/`   | YOLOv8 base weights and trained model outputs           |
| `outputs/` | Optional folder for saving results                      |
| `tests/`   | Minimal smoke tests to check the project layout         |
|------------|---------------------------------------------------------|


## Source Files
|-------------------------|-----------------------------------------------------|
| File                    | What it does                                        |
|-------------------------|-----------------------------------------------------|
| `src/settings.py`       | **All paths and hyperparameters — read this first** |
| `src/1_capture_data.py` | Open the webcam and save images for your dataset    |
| `src/2_split_data.py`   | Randomly split images into train / valid / test     |
| `src/3_train.py`        | Fine-tune YOLOv8 on your split dataset              |
| `src/4_inference.py`    | Load your trained model and run live detection      |
|-------------------------|-----------------------------------------------------|


## Setup with venv

From this `visual-ai-lab2/` directory:

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows Command Prompt:**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

## Alternative Conda Setup

```bash
conda env create -f environment.yml
conda activate visual-ai-lab2
```

---

## Running the Pipeline

Run the scripts from the `visual-ai-lab2/` directory in order:

```bash
# Step 1 — collect images (press 's' to save a frame, 'q' to quit)
python src/1_capture_data.py

# Step 2 — split into train / valid / test
python src/2_split_data.py

# Step 3 — train the model
python src/3_train.py

# Step 4 — run live detection with your trained model
python src/4_inference.py
```

---

## Tips

- **Start small** — use `EPOCHS = 10` in `settings.py` for your first run to verify the whole pipeline works, then increase to 50–100 for better accuracy.
- **Camera not detected?** Try setting `camera_index=1` inside the script.
- **Check training results** — after Step 3, open `model/my_model/results.png` to see loss and accuracy curves.
- **Tune the threshold** — if Step 4 shows too many false detections, raise `PRED_THRESHOLD` in `settings.py`. If it misses real objects, lower it.
- **Label quality matters** — bad or inconsistent labels are the most common cause of poor model performance.