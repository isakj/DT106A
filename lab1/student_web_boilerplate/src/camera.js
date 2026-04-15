(function setupCamera(globalObject) {
    const appNamespace = globalObject.VisualAILab1 || (globalObject.VisualAILab1 = {});

    class FrameSource {
        constructor(videoElement) {
            this.videoElement = videoElement;
            this.imageElement = null;
            this.stream = null;
            this.objectUrl = null;
            this.sourceType = "none";
            this.isStreaming = false;
        }

        async openWebcam() {
            this.stop();

            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false,
            });

            this.stream = stream;
            this.videoElement.srcObject = stream;
            this.videoElement.removeAttribute("loop");
            await this.videoElement.play();

            this.sourceType = "webcam";
            this.isStreaming = true;
        }

        async openFile(file) {
            this.stop();

            this.objectUrl = URL.createObjectURL(file);

            if (file.type.startsWith("image/")) {
                this.imageElement = await this.loadImage(this.objectUrl);
                this.sourceType = "image";
                this.isStreaming = false;
                return;
            }

            if (file.type.startsWith("video/")) {
                await this.loadVideo(this.objectUrl);
                this.videoElement.loop = true;
                await this.videoElement.play();
                this.sourceType = "video";
                this.isStreaming = true;
                return;
            }

            throw new Error("Unsupported file type. Please choose an image or video.");
        }

        async loadVideo(url) {
            this.videoElement.srcObject = null;
            this.videoElement.src = url;
            this.videoElement.load();

            await new Promise(function waitForMetadata(resolve, reject) {
                function cleanup() {
                    video.removeEventListener("loadedmetadata", onLoadedMetadata);
                    video.removeEventListener("error", onError);
                }

                function onLoadedMetadata() {
                    cleanup();
                    resolve();
                }

                function onError() {
                    cleanup();
                    reject(new Error("Could not load the selected video file."));
                }

                const video = this.videoElement;
                video.addEventListener("loadedmetadata", onLoadedMetadata);
                video.addEventListener("error", onError);
            }.bind(this));
        }

        loadImage(url) {
            return new Promise(function load(resolve, reject) {
                const image = new Image();
                image.onload = function handleLoad() {
                    resolve(image);
                };
                image.onerror = function handleError() {
                    reject(new Error("Could not load the selected image file."));
                };
                image.src = url;
            });
        }

        drawCurrentFrame(targetCanvas) {
            const context = targetCanvas.getContext("2d");

            if (this.sourceType === "image" && this.imageElement) {
                this.resizeCanvasToSource(targetCanvas, this.imageElement.naturalWidth, this.imageElement.naturalHeight);
                context.drawImage(this.imageElement, 0, 0, targetCanvas.width, targetCanvas.height);
                return true;
            }

            if ((this.sourceType === "webcam" || this.sourceType === "video") && this.videoElement.readyState >= 2) {
                this.resizeCanvasToSource(targetCanvas, this.videoElement.videoWidth, this.videoElement.videoHeight);
                context.drawImage(this.videoElement, 0, 0, targetCanvas.width, targetCanvas.height);
                return true;
            }

            return false;
        }

        resizeCanvasToSource(canvas, width, height) {
            if (!width || !height) {
                return;
            }

            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }
        }

        stop() {
            if (this.stream) {
                this.stream.getTracks().forEach(function stopTrack(track) {
                    track.stop();
                });
            }

            this.stream = null;
            this.imageElement = null;
            this.isStreaming = false;
            this.sourceType = "none";

            if (this.videoElement) {
                this.videoElement.pause();
                this.videoElement.removeAttribute("src");
                this.videoElement.srcObject = null;
                this.videoElement.load();
            }

            if (this.objectUrl) {
                URL.revokeObjectURL(this.objectUrl);
                this.objectUrl = null;
            }
        }
    }

    appNamespace.camera = {
        FrameSource,
    };
}(window));
