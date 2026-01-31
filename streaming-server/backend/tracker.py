"""
Object tracking module using YOLO
"""

import cv2
import time
from ultralytics import YOLO
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class ObjectTracker:
    """Handles YOLO object detection and tracking"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the object tracker
        
        Args:
            model_path: Path to YOLO model file
        """
        self.model_path = model_path or Config.YOLO_MODEL
        self.model = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, track: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process a frame with YOLO tracking
        
        Args:
            frame: Input frame (BGR format)
            track: Whether to use tracking (True) or just detection (False)
            
        Returns:
            Tuple of (annotated_frame, metadata)
        """
        self.frame_count += 1
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / (current_time - self.start_time)
            self.last_fps_update = current_time
        
        # Run YOLO inference
        if track:
            results = self.model.track(
                frame,
                persist=True,
                tracker=Config.TRACKER_CONFIG,
                verbose=False,
                conf=Config.CONFIDENCE_THRESHOLD
            )
        else:
            results = self.model(
                frame,
                verbose=False,
                conf=Config.CONFIDENCE_THRESHOLD
            )
        
        # Extract metadata
        metadata = self._extract_metadata(results[0])
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        return annotated_frame, metadata
    
    def _extract_metadata(self, result) -> Dict:
        """
        Extract detection metadata from YOLO result
        
        Args:
            result: YOLO result object
            
        Returns:
            Dictionary containing detection metadata
        """
        detections = []
        detected_classes = set()
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            track_ids = boxes.id.cpu().numpy().tolist() if boxes.id is not None else [None] * len(boxes)
            class_ids = boxes.cls.cpu().numpy().tolist()
            confidences = boxes.conf.cpu().numpy().tolist()
            bboxes = boxes.xyxy.cpu().numpy().tolist()  # [x1, y1, x2, y2]
            
            for i, class_id in enumerate(class_ids):
                class_name = self.model.names[int(class_id)]
                detected_classes.add(class_name)
                
                detection = {
                    "track_id": int(track_ids[i]) if track_ids[i] is not None else None,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": round(confidences[i], 3),
                    "bbox": [round(coord, 2) for coord in bboxes[i]]
                }
                detections.append(detection)
        
        metadata = {
            "num_detections": len(detections),
            "detections": detections,
            "detected_classes": list(sorted(detected_classes)),
            "fps": round(self.fps, 1),
            "frame_count": self.frame_count
        }
        
        return metadata
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def reset_stats(self):
        """Reset frame count and timing stats"""
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0


class CameraCapture:
    """Handles camera capture with configuration"""
    
    def __init__(self, camera_index: int = None, width: int = None, height: int = None):
        """
        Initialize camera capture
        
        Args:
            camera_index: Camera device index
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index or Config.CAMERA_INDEX
        self.width = width or Config.CAMERA_WIDTH
        self.height = height or Config.CAMERA_HEIGHT
        self.cap = None
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with settings"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Verify resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera initialized: {actual_width}x{actual_height}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from camera"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()
