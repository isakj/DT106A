# Laboration 2 — From Raw Images to a Trained Object Detector

## Objective

Build and evaluate a complete object detection pipeline using your own collected data.  
The goal is to understand how a visual AI system is constructed end-to-end:

**data collection → annotation → training → inference**

This lab focuses on the full machine learning workflow: what data you need, how you prepare it, how a model learns from it, and where it breaks down.

---

## Task

Start from the provided project folder structure. It contains numbered scripts that guide you
through each stage of the pipeline. Extend these scripts to build your own detector.

### Stage 1 — Capture Data

Use the provided `1_capture_data.py` as a starting point.

- Capture images of **two distinct object classes** using your webcam
  - If your webcam does not work, use a smartphone camera and transfer images manually
- Collect at least **30 images per class** under varied conditions (lighting, angle, distance)
- Save all raw images to the `dataset/images/` folder

### Stage 2 — Annotate

Annotate your images using a bounding-box annotation tool (e.g., [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg)).

- Draw one bounding box per object instance
- Export annotations in **YOLO format** (one `.txt` file per image)
- Save label files to `dataset/labels/` with the same filenames as the images
- Update `dataset/dataset.yaml` with your class names

### Stage 3 — Split Data

Run `2_split_data.py` to split your dataset into train / validation / test sets.

- Use the default split ratios (70% / 10% / 20%) or adjust them in `settings.py`
- Verify that all three splits contain images from both classes

### Stage 4 — Train

Run `3_train.py` to fine-tune a pre-trained YOLOv8 model on your dataset.

- Start from the provided `yolov8n.pt` base model
- Train for at least 50 epochs (default: 100)
- Monitor the loss and metric curves saved in the `model/` folder after training

### Stage 5 — Inference

Run `4_inference.py` to test your trained model live via webcam.

- Your running application must:
  - Draw bounding boxes with class labels and confidence scores
  - Display a detection count overlay
  - Allow the confidence threshold to be adjusted at runtime using keyboard input (`+` / `-`)
  - Run continuously until manually stopped (`q` or `Esc`)

---

## Implementation Requirements

- Keep the numbered script structure — one script per pipeline stage
- Place shared configuration (paths, hyperparameters) in `settings.py` only
- Do not hardcode paths inside scripts; resolve them relative to `settings.py`

---

## Analysis Requirements

You should demonstrate and discuss:

- **One scenario where detection works well**  
  What conditions made it reliable? What does the confidence score tell you?

- **One scenario where detection fails or behaves poorly**  
  What caused the failure — data quality, class imbalance, lighting, occlusion, something else?

For both cases, include visual evidence (screenshots or captured frames) and explain what
the model is doing and why.

Additionally, reflect on:

- How many images were enough? What would more data have changed?
- What do the training curves (loss, precision, recall) tell you about your training run?

---

## Optional Extension (choose one)

- Add a third class and retrain
- Implement confidence-based color coding on bounding boxes (e.g., green = high, red = low)
- Support loading a saved image or video file as input instead of live webcam
- Export a short annotated video clip showing your detector running
- Compare two training runs with different numbers of epochs or data sizes

---

## Submission Format (5 slides presented as a PDF)

1. **Goal and Setup**  
   What you built, what your two classes are, and how the pipeline was run

2. **Data and Annotation**  
   How many images, how they were collected, and how they were annotated

3. **Training Results**  
   Loss and metric curves — what they show about how your model learned

4. **Detection Results**  
   One success case and one failure case with visual evidence

5. **Analysis and Takeaways**  
   Why it works, why it fails, and what this lab shows about data-driven vision systems
