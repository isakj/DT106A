(function setupSettings(globalObject) {
    const appNamespace = globalObject.VisualAILab1 || (globalObject.VisualAILab1 = {});

    appNamespace.settings = {
        appTitle: "Visual AI Lab 1 - Web Boilerplate",
        defaultSource: "webcam",
    };
}(window));
