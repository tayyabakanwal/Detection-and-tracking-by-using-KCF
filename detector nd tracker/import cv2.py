import cv2
import numpy as np
from ultralytics import YOLO

class EnhancedSTCTracker:
    def __init__(self, frame, bbox):
        x, y, w, h = bbox
        self.bbox = (x, y, w, h)
        self.template = frame[y:y+h, x:x+w]
        self.method = cv2.TM_CCOEFF_NORMED
        self.threshold = 0.5 # Confidence threshold for reinitializing the tracker
        self.scale_factors = [0.9, 1.0, 1.1]  # Different scales for multi-scale matching

    def update(self, frame):
        best_val = -1
        best_bbox = self.bbox

        for scale in self.scale_factors:
            resized_template = cv2.resize(self.template, (0, 0), fx=scale, fy=scale)
            result = cv2.matchTemplate(frame, resized_template, self.method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                x, y = max_loc
                w, h = int(resized_template.shape[1] / scale), int(resized_template.shape[0] / scale)
                best_bbox = (x, y, w, h)

        if best_val < self.threshold:
            return False, self.bbox

        self.bbox = best_bbox
        x, y, w, h = self.bbox
        self.template = frame[y:y+h, x:x+w]

        return True, self.bbox

def main():
    # Load the YOLOv8 model
    model = YOLO('C:\\Users\\hp\\Desktop\\tyba\\train39\\tom.pt')  # Ensure this is the correct model file

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Initialize tracking variables
    tracking = False
    tracker = None

    # Define the class name to filter
    target_class_name = 'tom'

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if not tracking:
            # Make predictions
            results = model(frame)

            # Extract detection details
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]
                    label = result.names[class_id]  # Get the class name

                    if label == target_class_name:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        tracker = EnhancedSTCTracker(frame, bbox)
                        tracking = True
                        break

        if tracking:
            success, bbox = tracker.update(frame)
            if success:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, f"Class: {target_class_name}", (p1[0], p1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (p1[0], p1[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            else:
                tracking = False
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the frame with bounding boxes
        cv2.imshow('YOLOv8 Detection and Tracking', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
