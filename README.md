# Detection-and-tracking-by-using-KCFThis Python script performs object detection and tracking using the YOLOv8 model and an enhanced template matching-based tracking algorithm. Here is an overview of the key components and functionality:

### 1. **EnhancedSTCTracker Class**
The `EnhancedSTCTracker` class is an enhanced template matching tracker that:
- **Initialization**: Takes an initial frame and bounding box (`bbox`) of the object to be tracked. It extracts a template from the given `bbox` and sets parameters like the matching method and confidence threshold.
- **Update Method**: In each frame, the tracker performs multi-scale template matching to find the best match location. If the best match confidence is below the threshold, it indicates a tracking failure.

### 2. **Main Function**
The `main` function handles the object detection and tracking workflow:
- **Model Loading**: Loads the YOLOv8 model for object detection.
- **Webcam Capture**: Opens the webcam for real-time video capture.
- **Tracking Initialization**: Checks if tracking is active. If not, it uses the YOLOv8 model to detect objects in the current frame.
- **Detection**: Iterates over the detected objects, checking if the detected object's class matches the target class name ('tom'). If a match is found, it initializes the tracker.
- **Tracking**: Updates the tracker with the current frame. If tracking is successful, it draws a bounding box around the tracked object and displays the class name and confidence. If tracking fails, it resets the tracking state.
- **Display**: Displays the video frame with annotations.
- **Exit Condition**: Exits the loop and releases resources if the 'q' key is pressed.

### Dependencies
- **OpenCV (cv2)**: For image processing and video capture.
- **NumPy**: For numerical operations.
- **YOLO**: For object detection using the YOLOv8 model.

### Usage
1. Ensure you have the required dependencies installed.
2. Provide the correct path to your YOLOv8 model (`tom.pt`).
3. Run the script. It will open the webcam and start detecting and tracking objects of the specified class.
   
