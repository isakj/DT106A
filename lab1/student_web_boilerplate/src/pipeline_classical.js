(function setupPipeline(globalObject) {
    const appNamespace = globalObject.VisualAILab1 || (globalObject.VisualAILab1 = {});

    function processFrame(inputCanvas, outputCanvas, options) {
        if (!globalObject.cv || !globalObject.cv.imread) {
            throw new Error("OpenCV.js is not ready yet.");
        }

        const cv = globalObject.cv;
        const utils = appNamespace.utils;

        const src = cv.imread(inputCanvas);
        const gray = new cv.Mat();
        const display = new cv.Mat();

        try {
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            cv.cvtColor(gray, display, cv.COLOR_GRAY2RGBA);

            cv.imshow(outputCanvas, display);

            return {
                mode: "grayscale_demo",
                note: "TODO: add blur, structure extraction, extra mode, and runtime parameter",
                description: utils.describePipeline(),
            };
        } finally {
            src.delete();
            gray.delete();
            display.delete();
        }
    }

    appNamespace.pipeline = {
        processFrame,
    };
}(window));
