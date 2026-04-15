# Student Web Boilerplate: Visual AI Lab 1

This folder is a browser-based alternative to the Python Lab 1 starter.

It mirrors the same pipeline idea:

`input -> processing -> output`

The goal is to let students complete the same Lab 1 reasoning with HTML, JavaScript, Canvas, and OpenCV.js when a local Python environment is not practical.

Some parts of the code are intentionally incomplete. Look for `TODO` markers and extend the scaffold during the lab.

## Folder Overview

- `index.html`
  Main page for the lab app.
- `styles.css`
  Lightweight layout and presentation styles.
- `src/main.js`
  Connects input, processing, and output.
- `src/camera.js`
  Handles webcam, image, and video sources.
- `src/pipeline_classical.js`
  Contains the classical pipeline that students can extend.
- `src/ui.js`
  Draws overlays and updates control labels.
- `src/settings.js`
  Stores simple configuration values.
- `src/utils.js`
  Optional helpers shared across modules.

## How This Mirrors The Python Track

The module boundaries are intentionally parallel to the Python version:

- `main.py` <-> `main.js`
- `camera.py` <-> `camera.js`
- `pipeline_classical.py` <-> `pipeline_classical.js`
- `ui.py` <-> `ui.js`
- `settings.py` <-> `settings.js`
- `utils.py` <-> `utils.js`

The same teaching target applies in both tracks:

- select an input source,
- process frames in a small pipeline,
- support multiple processing modes,
- adjust one parameter during runtime,
- show original and processed output,
- overlay useful status information.

This starter does not implement the full task for you. It only demonstrates the module boundaries, source handling, side-by-side display, and one minimal grayscale processing path.

## Running The Browser Track

1. Open `index.html` in a modern browser.
2. Wait for OpenCV.js to finish loading.
3. Start with webcam input if the browser allows camera access.
4. If webcam access is blocked or unavailable, upload an image or video file instead.

## Controls

- `Start webcam`
  Request live camera input.
- `Upload image/video`
  Load a fallback file input from disk.
- `Stop source`
  Stop the current webcam or file source.

## Suggested Student Work

Students should treat `src/pipeline_classical.js` as the main extension point.

Suggested tasks:

- replace the grayscale-only demo with a three-stage pipeline,
- add at least two processing modes,
- add one parameter that can be changed during runtime,
- change what information is shown in the overlay,
- document one success case and one failure case for the final submission.

## Browser Notes

- Webcam access depends on browser permissions and security policy.
- Image and video upload are the intended fallback path when webcam access is not available.
- This scaffold loads OpenCV.js from the official OpenCV documentation CDN, so an internet connection is required the first time students use this track.
- The fuller browser implementation is kept separate in the private instructor web reference, not in this student starter.

## Official Lab Instructions

The official Lab 1 instructions are provided on the course Canvas page. This folder mirrors those instructions with a browser-based implementation path, but Canvas remains the authoritative source for the objective, task, analysis requirements, optional extension, and submission format.
