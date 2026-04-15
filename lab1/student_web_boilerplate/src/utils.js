(function setupUtils(globalObject) {
    const appNamespace = globalObject.VisualAILab1 || (globalObject.VisualAILab1 = {});

    function describePipeline() {
        return "grayscale only for now";
    }

    function clearCanvas(canvas, message) {
        const context = canvas.getContext("2d");
        context.fillStyle = "#0b0f14";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "#ffffff";
        context.font = "20px Arial";
        context.fillText(message, 20, 40);
    }

    appNamespace.utils = {
        describePipeline,
        clearCanvas,
    };
}(window));
