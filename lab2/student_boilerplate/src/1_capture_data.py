import cv2
import os
from datetime import datetime
from settings import CWDIR

class DataCollector:
    def __init__(self, save_dir=os.path.join(CWDIR, "../../dataset/org_data"), camera_index=0):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.camera_index = camera_index
        self.cap = None
        self.image_count = 0

    def open_camera(self):
        print("[INFO] Opening camera...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera (index={self.camera_index}). Try index 1."
            )

        print("[INFO] Camera opened successfully")

    def generate_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"img_{timestamp}.jpg"

    def save_frame(self, frame):

        """Complete this function to save the current frame to the specified directory with a unique filename. You can use the generate_filename method to create a unique name for each image.
        TODO:
        - Generate a unique filename for the image using the generate_filename method.
        - Save the image to the save_dir using cv2.imwrite.
        - Increment the image_count and print a message indicating the image has been saved along with the total count of saved images.
        """
        return

    def draw_overlay(self, frame):
        text1 = "Press 's' to save | 'q' to quit"
        text2 = f"Saved images: {self.image_count}"

        cv2.putText(
            frame,
            text1,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            text2,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return frame

    def run(self):
        try:
            self.open_camera()
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return

        print("[INFO] Press 's' to save images | 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Keep original frame for saving
            raw_frame = frame.copy()

            # Create a separate frame for display
            display_frame = self.draw_overlay(frame)

            cv2.imshow("Lab 2 - Data Collector", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("[INFO] Exiting...")
                break

            if key == ord("s"):
                self.save_frame(raw_frame)  # Save image

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()