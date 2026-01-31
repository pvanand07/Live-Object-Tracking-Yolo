"""
VisionEye Object Tracking with Event Video Saver
Saves video clips when objects are detected using Ultralytics VisionEye solution
Organizes recordings by class and timestamp
"""

import cv2
from ultralytics import solutions
from datetime import datetime
from pathlib import Path
import json


class EventVideoSaver:
    """Handles saving video clips when events (detections) occur"""
    
    def __init__(self, base_dir="events", recording_buffer_seconds=5, fps=20):
        """
        Initialize the video saver
        
        Args:
            base_dir: Base directory for saving event videos
            recording_buffer_seconds: Seconds to continue recording after last detection
            fps: Frames per second for saved videos
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.recording_buffer_seconds = recording_buffer_seconds
        self.fps = fps
        
        self.is_recording = False
        self.video_writer = None
        self.current_event_dir = None
        self.last_detection_time = None
        self.detected_classes = set()
        self.frame_size = None
        self.event_start_time = None
        self.frame_count = 0
        self.class_detection_counts = {}
        self.frame_metadata = []  # Store per-frame metadata
        
    def start_recording(self, frame, detected_classes):
        """Start a new recording for an event"""
        if self.is_recording:
            return
            
        # Create timestamp and directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        classes_str = "_".join(sorted(detected_classes))
        event_name = f"{classes_str}_{timestamp}"
        
        # Create event directory
        self.current_event_dir = self.base_dir / event_name
        self.current_event_dir.mkdir(exist_ok=True)
        
        # Setup video writer
        video_path = self.current_event_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps,
            self.frame_size
        )
        
        self.is_recording = True
        self.detected_classes = detected_classes.copy()
        self.event_start_time = datetime.now()
        self.frame_count = 0
        self.class_detection_counts = {cls: 0 for cls in detected_classes}
        self.frame_metadata = []  # Reset frame metadata for new recording
        print(f"📹 Started recording event: {event_name}")
        
    def write_frame(self, frame, frame_info=None):
        """Write a frame to the current recording"""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame)
            self.frame_count += 1
            
            # Store per-frame metadata
            if frame_info is not None:
                self.frame_metadata.append(frame_info)
            
    def stop_recording(self):
        """Stop the current recording"""
        if not self.is_recording:
            return
            
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save metadata
        self._save_metadata()
            
        print(f"⏹️  Stopped recording: {self.current_event_dir.name}")
        self.is_recording = False
        self.current_event_dir = None
        self.detected_classes = set()
        self.frame_count = 0
        self.class_detection_counts = {}
        self.frame_metadata = []
        
    def should_stop_recording(self):
        """Check if recording should stop based on buffer time"""
        if not self.is_recording or self.last_detection_time is None:
            return False
            
        elapsed = (datetime.now() - self.last_detection_time).total_seconds()
        return elapsed > self.recording_buffer_seconds
        
    def update(self, frame, detections_present, detected_classes, frame_info=None):
        """
        Update recording state based on detections
        
        Args:
            frame: Current frame
            detections_present: Boolean indicating if objects were detected
            detected_classes: Set of detected class names
            frame_info: Optional dict containing per-frame detection details
        """
        if detections_present:
            self.last_detection_time = datetime.now()
            
            # Add new classes to the set
            self.detected_classes.update(detected_classes)
            
            # Update detection counts
            for cls in detected_classes:
                if cls not in self.class_detection_counts:
                    self.class_detection_counts[cls] = 0
                self.class_detection_counts[cls] += 1
            
            # Start recording if not already
            if not self.is_recording:
                self.start_recording(frame, detected_classes)
        
        # Write frame if recording
        if self.is_recording:
            self.write_frame(frame, frame_info)
            
            # Check if should stop
            if self.should_stop_recording():
                self.stop_recording()
                
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
    
    def _save_metadata(self):
        """Save event metadata to JSON file"""
        if self.current_event_dir is None:
            return
        
        event_end_time = datetime.now()
        duration = (event_end_time - self.event_start_time).total_seconds()
        
        metadata = {
            "event_name": self.current_event_dir.name,
            "start_time": self.event_start_time.isoformat(),
            "end_time": event_end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "detected_classes": list(sorted(self.detected_classes)),
            "class_detection_counts": self.class_detection_counts,
            "total_frames": self.frame_count,
            "fps": self.fps,
            "frame_size": self.frame_size,
            "video_file": "video.mp4",
            "frames": self.frame_metadata  # Per-frame metadata
        }
        
        metadata_path = self.current_event_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved metadata: {metadata_path.name}")


def main():
    """Main function to run VisionEye tracking with event video saving"""
    
    # Configuration
    RECORDING_BUFFER_SECONDS = 5  # Continue recording for 5 seconds after last detection
    VIDEO_SOURCE = 0  # Use webcam (0) or provide path to video file
    TRACK_CLASSES = [0, 2]  # Class IDs to track (0=person, 2=car by default in COCO)
    MODEL = "yolov8n.pt"  # YOLO model to use
    
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    assert cap.isOpened(), "Error reading video source"
    
    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    
    # If using webcam, fps might be 0, set a default
    if fps == 0:
        fps = 20
    
    print(f"Video properties: {w}x{h} @ {fps} FPS")
    
    # Initialize VisionEye object
    print("Initializing VisionEye...")
    visioneye = solutions.VisionEye(
        show=True,  # display the output
        model=MODEL,
        classes=TRACK_CLASSES,  # track specific classes
    )
    
    # Initialize event video saver
    saver = EventVideoSaver(
        base_dir="events",
        recording_buffer_seconds=RECORDING_BUFFER_SECONDS,
        fps=fps
    )
    
    print("Starting VisionEye tracking with event recording... Press 'q' to quit")
    
    try:
        while cap.isOpened():
            success, im0 = cap.read()
            
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            
            # Process frame with VisionEye
            results = visioneye(im0)
            
            # Check for detections and extract class names + per-frame metadata
            detections_present = False
            detected_classes = set()
            frame_info = None
            
            # VisionEye stores tracking results internally in visioneye.tracks
            # Access the tracks directly from the VisionEye object
            if hasattr(visioneye, 'tracks') and visioneye.tracks is not None:
                if hasattr(visioneye.tracks, 'boxes') and visioneye.tracks.boxes is not None:
                    boxes = visioneye.tracks.boxes
                    if hasattr(boxes, 'cls') and len(boxes.cls) > 0:
                        detections_present = True
                        
                        # Extract detailed per-frame information
                        track_ids = boxes.id.cpu().numpy().tolist() if boxes.id is not None else []
                        class_ids = boxes.cls.cpu().numpy().tolist()
                        confidences = boxes.conf.cpu().numpy().tolist()
                        bboxes = boxes.xyxy.cpu().numpy().tolist()  # [x1, y1, x2, y2]
                        
                        # Build per-frame metadata
                        detections = []
                        for i, class_id in enumerate(class_ids):
                            # Get class name from the model
                            if hasattr(visioneye, 'model') and hasattr(visioneye.model, 'names'):
                                class_name = visioneye.model.names[int(class_id)]
                            else:
                                class_name = f"class_{int(class_id)}"
                            
                            detected_classes.add(class_name)
                            
                            detection = {
                                "track_id": int(track_ids[i]) if i < len(track_ids) else None,
                                "class_id": int(class_id),
                                "class_name": class_name,
                                "confidence": round(confidences[i], 3),
                                "bbox": [round(coord, 2) for coord in bboxes[i]]  # [x1, y1, x2, y2]
                            }
                            detections.append(detection)
                        
                        frame_info = {
                            "frame_number": saver.frame_count + 1 if saver.is_recording else 0,
                            "timestamp": datetime.now().isoformat(),
                            "num_detections": len(detections),
                            "detections": detections
                        }
            
            # Get the plotted frame from VisionEye
            output_frame = results.plot_im if hasattr(results, 'plot_im') else im0
            
            # Add recording indicator to the frame
            if saver.is_recording:
                cv2.putText(
                    output_frame,
                    '🔴 RECORDING',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                
                # Show detected classes
                classes_text = f"Classes: {', '.join(sorted(saver.detected_classes))}"
                cv2.putText(
                    output_frame,
                    classes_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
            
            # Update video saver with frame metadata
            saver.update(output_frame, detections_present, detected_classes, frame_info)
            
            # The VisionEye solution already shows the output if show=True
            # If you want manual control, set show=False and use cv2.imshow here
            # cv2.imshow('VisionEye - Event Saver', output_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Cleanup
        saver.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("VisionEye tracking stopped")


if __name__ == "__main__":
    main()
