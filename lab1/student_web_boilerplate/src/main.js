(function setupMain(globalObject) {
    const appNamespace = globalObject.VisualAILab1 || (globalObject.VisualAILab1 = {});

    const state = {
        cvReady: false,
        animationFrameId: null,
    };

    let elements;
    let frameSource;

    function getStatusSnapshot() {
        return {
            source: frameSource ? frameSource.sourceType : "none",
            mode: "grayscale_demo",
            note: state.cvReady
                ? "TODO: extend this starter with more modes, parameters, and pipeline steps."
                : "Waiting for OpenCV.js to load.",
        };
    }

    function renderPlaceholder() {
        appNamespace.utils.clearCanvas(elements.originalCanvas, "Waiting for input");
        appNamespace.utils.clearCanvas(elements.processedCanvas, "Waiting for input");
        appNamespace.ui.updatePipelineStatus(elements, getStatusSnapshot());
    }

    function stopRenderLoop() {
        if (state.animationFrameId !== null) {
            cancelAnimationFrame(state.animationFrameId);
            state.animationFrameId = null;
        }
    }

    function processAndDraw() {
        if (!frameSource.drawCurrentFrame(elements.originalCanvas)) {
            if (frameSource.sourceType !== "none") {
                appNamespace.ui.setAppMessage(elements, "The selected source has no available frame yet.", false);
            }
            return false;
        }

        let status;

        try {
            status = appNamespace.pipeline.processFrame(
                elements.originalCanvas,
                elements.processedCanvas,
                {}
            );
        } catch (error) {
            appNamespace.ui.setAppMessage(elements, error.message, true);
            return false;
        }

        status.source = frameSource.sourceType;
        appNamespace.ui.drawStatusOverlay(elements.processedCanvas, status);
        appNamespace.ui.updatePipelineStatus(elements, status);
        return true;
    }

    function tick() {
        processAndDraw();

        if (frameSource.isStreaming) {
            state.animationFrameId = requestAnimationFrame(tick);
        } else {
            state.animationFrameId = null;
        }
    }

    function refreshOutput() {
        if (!state.cvReady) {
            return;
        }

        stopRenderLoop();

        if (frameSource.isStreaming) {
            tick();
            return;
        }

        if (!processAndDraw()) {
            renderPlaceholder();
        }
    }

    async function startWebcam() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error("This browser does not support webcam access.");
            }

            await frameSource.openWebcam();
            appNamespace.ui.setAppMessage(elements, "Webcam started. If camera access fails, upload an image or video instead.", false);
            refreshOutput();
        } catch (error) {
            appNamespace.ui.setAppMessage(elements, error.message, true);
            renderPlaceholder();
        }
    }

    async function handleFileSelection(event) {
        const file = event.target.files[0];
        if (!file) {
            return;
        }

        try {
            await frameSource.openFile(file);
            appNamespace.ui.setAppMessage(elements, "Fallback file loaded. You can switch back to webcam at any time.", false);
            refreshOutput();
        } catch (error) {
            appNamespace.ui.setAppMessage(elements, error.message, true);
            renderPlaceholder();
        }
    }

    function stopSource() {
        stopRenderLoop();
        frameSource.stop();
        renderPlaceholder();
        appNamespace.ui.setAppMessage(elements, "Source stopped. Start webcam or upload an image/video file.", false);
    }

    function waitForOpenCv() {
        if (!globalObject.cv) {
            globalObject.setTimeout(waitForOpenCv, 100);
            return;
        }

        globalObject.cv.onRuntimeInitialized = function onRuntimeInitialized() {
            state.cvReady = true;
            appNamespace.ui.setAppMessage(elements, "OpenCV.js loaded. Start with webcam, or upload an image/video file.", false);
            renderPlaceholder();
        };
    }

    function bindEvents() {
        elements.startWebcamButton.addEventListener("click", startWebcam);
        elements.stopSourceButton.addEventListener("click", stopSource);
        elements.fileInput.addEventListener("change", handleFileSelection);
    }

    function initialize() {
        elements = appNamespace.ui.getElements();
        frameSource = new appNamespace.camera.FrameSource(elements.videoElement);
        bindEvents();
        renderPlaceholder();
        waitForOpenCv();
    }

    document.addEventListener("DOMContentLoaded", initialize);
}(window));
