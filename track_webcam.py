"""
Object Tracking with YOLO and Webcam
Uses Ultralytics YOLO for real-time object tracking from webcam feed
"""

import cv2
from ultralytics import YOLO


def main():
    """Main function to run object tracking on webcam feed"""
    
    # Load YOLOv8 model (will download automatically on first run)
    # You can use: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), etc.
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # Using nano model for faster performance
    
    # Initialize webcam (0 is usually the default camera)
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting object tracking... Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        success, frame = cap.read()
        
        if not success:
            print("Error: Failed to read frame")
            break
        
        # Perform tracking on the frame
        # persist=True maintains tracking IDs across frames
        # tracker can be: 'botsort.yaml' or 'bytetrack.yaml'
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display tracking info
        if results[0].boxes.id is not None:
            num_objects = len(results[0].boxes.id)
            cv2.putText(
                annotated_frame,
                f'Tracking {num_objects} objects',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Show the annotated frame
        cv2.imshow('YOLO Object Tracking', annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped")


if __name__ == "__main__":
    main()
