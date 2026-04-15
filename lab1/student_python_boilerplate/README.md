# Student Boilerplate: Visual AI Lab 1

This folder is your starting point for Lab 1.

In this lab, you will work with a simple vision pipeline:

`input -> processing -> output`

Some parts of the code are intentionally incomplete. Look for `TODO` markers and fill them in during the lab.

## Folder Overview

- `src/`
  Main source files for the lab.
- `offline_inputs/test_images/`
  Place image files here if you want to test without a webcam.
- `offline_inputs/test_videos/`
  Place video files here if webcam input is unavailable.
- `outputs/`
  Optional output folder for saved results.
- `tests/`
  Minimal placeholder for simple tests.

## Main Source Files

- `src/main.py`
  Connects input, processing, and output.
- `src/camera.py`
  Handles webcam, image, or video input.
- `src/pipeline_classical.py`
  Contains the classical pipeline you will extend.
- `src/ui.py`
  Draws simple overlay text.
- `src/settings.py`
  Stores a few configuration values.
- `src/utils.py`
  Optional helper file if you need shared utility code.

## Setup with venv

From this `student_python_boilerplate/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Alternative Conda Setup

```bash
conda env create -f environment.yml
conda activate visual-ai-lab1
```

## Emergency Fallback

If environment creation fails, try:

```bash
pip install numpy opencv-python
```

## Running the Lab

Use the webcam:

```bash
python src/main.py
```

Use an image file:

```bash
python src/main.py --source image --path offline_inputs/test_images/example.jpg
```

Use a video file:

```bash
python src/main.py --source video --path offline_inputs/test_videos/example.mp4
```

## Notes

- The boilerplate is intentionally minimal.
- The default pipeline is intentionally simple.
- If the webcam does not work on your machine, use image or video input instead.
