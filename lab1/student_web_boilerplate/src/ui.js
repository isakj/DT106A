(function setupUi(globalObject) {
    const appNamespace = globalObject.VisualAILab1 || (globalObject.VisualAILab1 = {});

    function getElements() {
        return {
            startWebcamButton: document.getElementById("start-webcam"),
            stopSourceButton: document.getElementById("stop-source"),
            fileInput: document.getElementById("file-input"),
            appMessage: document.getElementById("app-message"),
            sourceValue: document.getElementById("source-value"),
            modeValue: document.getElementById("mode-value"),
            noteValue: document.getElementById("note-value"),
            originalCanvas: document.getElementById("original-canvas"),
            processedCanvas: document.getElementById("processed-canvas"),
            videoElement: document.getElementById("video-element"),
        };
    }

    function setAppMessage(elements, message, isError) {
        elements.appMessage.textContent = message;
        elements.appMessage.style.color = isError ? "#b42318" : "#46515c";
    }

    function updatePipelineStatus(elements, status) {
        elements.sourceValue.textContent = status.source;
        elements.modeValue.textContent = status.mode;
        elements.noteValue.textContent = status.note;
    }

    function drawStatusOverlay(canvas, status) {
        const context = canvas.getContext("2d");
        const lines = [
            "Mode: " + status.mode,
            status.note,
        ];

        context.save();
        context.fillStyle = "rgba(0, 0, 0, 0.6)";
        context.fillRect(10, 10, canvas.width * 0.58, 20 + (lines.length * 22));
        context.fillStyle = "#ffffff";
        context.font = "18px Arial";

        lines.forEach(function renderLine(line, index) {
            context.fillText(line, 20, 35 + (index * 20));
        });

        context.restore();
    }

    appNamespace.ui = {
        getElements,
        setAppMessage,
        updatePipelineStatus,
        drawStatusOverlay,
    };
}(window));
