"""
FastAPI WebSocket Video Streaming Server
Streams video with YOLO object detection to web clients
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import cv2
import base64
import asyncio
import json
from pathlib import Path
from typing import Set, List, Dict
import sys
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from backend.tracker import ObjectTracker, CameraCapture


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
            
        # Create unique event ID using UUID
        event_id = str(uuid.uuid4())
        
        # Create event directory with UUID
        self.current_event_dir = self.base_dir / event_id
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
        print(f"📹 Started recording event: {event_id}")
        
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
            "event_id": self.current_event_dir.name,
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

# Initialize FastAPI app
app = FastAPI(title="Object Tracking Stream", version="1.0.0")

# Global tracker, camera, and event saver
tracker: ObjectTracker = None
camera: CameraCapture = None
event_saver: EventVideoSaver = None
active_connections: Set[WebSocket] = set()

# Mount static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize tracker, camera, and event saver on startup"""
    global tracker, camera, event_saver
    
    print("\n" + "="*60)
    print("🚀 Starting Object Tracking Streaming Server")
    print("="*60)
    
    try:
        # Initialize tracker
        tracker = ObjectTracker()
        
        # Initialize camera
        camera = CameraCapture()
        
        # Initialize event video saver
        event_saver = EventVideoSaver(
            base_dir="events",
            recording_buffer_seconds=5,  # Continue recording for 5 seconds after last detection
            fps=20  # Save video at 20 fps
        )
        
        print("\n✓ Server initialization complete")
        print(f"✓ Configuration: {Config.get_info()}")
        print(f"✓ Event recording enabled (events/ directory)")
        print(f"\n📡 Server running at http://{Config.HOST}:{Config.PORT}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera, event_saver, active_connections
    
    print("\n🛑 Shutting down server...")
    
    # Close all active connections
    for connection in active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    active_connections.clear()
    
    # Cleanup event saver
    if event_saver:
        event_saver.cleanup()
    
    # Release camera
    if camera:
        camera.release()
    
    print("✓ Cleanup complete\n")


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    html_file = frontend_path / "index.html"
    return FileResponse(html_file)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "camera_opened": camera.is_opened() if camera else False,
        "active_connections": len(active_connections),
        "config": Config.get_info()
    }


@app.get("/api/events")
async def list_events() -> List[Dict]:
    """List all recorded events"""
    if not event_saver:
        raise HTTPException(status_code=503, detail="Event saver not initialized")
    
    events_dir = event_saver.base_dir
    if not events_dir.exists():
        return []
    
    events = []
    for event_dir in events_dir.iterdir():
        if event_dir.is_dir():
            metadata_file = event_dir / "metadata.json"
            video_file = event_dir / "video.mp4"
            
            # Load metadata if available
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    events.append({
                        "event_id": event_dir.name,
                        "event_name": metadata.get("event_name", event_dir.name),
                        "start_time": metadata.get("start_time"),
                        "end_time": metadata.get("end_time"),
                        "duration_seconds": metadata.get("duration_seconds"),
                        "detected_classes": metadata.get("detected_classes", []),
                        "total_frames": metadata.get("total_frames", 0),
                        "has_video": video_file.exists(),
                        "has_metadata": True
                    })
            else:
                # Event directory exists but no metadata yet (recording in progress)
                events.append({
                    "event_id": event_dir.name,
                    "event_name": event_dir.name,
                    "start_time": None,
                    "end_time": None,
                    "duration_seconds": None,
                    "detected_classes": [],
                    "total_frames": 0,
                    "has_video": video_file.exists(),
                    "has_metadata": False,
                    "recording_in_progress": True
                })
    
    # Sort by start time (most recent first)
    events.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    return events


@app.get("/api/events/{event_id}/metadata")
async def get_event_metadata(event_id: str) -> Dict:
    """Get metadata for a specific event"""
    if not event_saver:
        raise HTTPException(status_code=503, detail="Event saver not initialized")
    
    event_dir = event_saver.base_dir / event_id
    if not event_dir.exists():
        raise HTTPException(status_code=404, detail="Event not found")
    
    metadata_file = event_dir / "metadata.json"
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Event metadata not found")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata


@app.get("/api/events/{event_id}/video")
async def stream_event_video(event_id: str):
    """Stream video for a specific event"""
    if not event_saver:
        raise HTTPException(status_code=503, detail="Event saver not initialized")
    
    event_dir = event_saver.base_dir / event_id
    if not event_dir.exists():
        raise HTTPException(status_code=404, detail="Event not found")
    
    video_file = event_dir / "video.mp4"
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Event video not found")
    
    def iterfile():
        with open(video_file, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(iterfile(), media_type="video/mp4")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming video frames
    """
    global active_connections
    
    await websocket.accept()
    active_connections.add(websocket)
    
    print(f"✓ New WebSocket connection (Total: {len(active_connections)})")
    
    try:
        # Send initial config
        await websocket.send_json({
            "type": "config",
            "data": Config.get_info()
        })
        
        while True:
            # Check if camera is available
            if not camera or not camera.is_opened():
                await websocket.send_json({
                    "type": "error",
                    "message": "Camera not available"
                })
                await asyncio.sleep(1)
                continue
            
            # Read frame from camera
            success, frame = camera.read()
            
            if not success:
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to read frame"
                })
                await asyncio.sleep(0.1)
                continue
            
            # Process frame with YOLO tracking
            annotated_frame, metadata = tracker.process_frame(frame, track=True)
            
            # Extract detection information for event saver
            detections_present = metadata.get('num_detections', 0) > 0
            detected_classes = set()
            frame_info = None
            
            if detections_present and 'detections' in metadata:
                # Extract class names
                for detection in metadata['detections']:
                    if 'class_name' in detection:
                        detected_classes.add(detection['class_name'])
                
                # Build per-frame metadata for event recording
                frame_info = {
                    "frame_number": event_saver.frame_count + 1 if event_saver.is_recording else 0,
                    "timestamp": datetime.now().isoformat(),
                    "num_detections": len(metadata['detections']),
                    "detections": metadata['detections']
                }
            
            # Update event saver with original frame (not annotated)
            event_saver.update(frame, detections_present, detected_classes, frame_info)
            
            # Encode frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
            success, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            
            if not success:
                continue
            
            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Add recording status to metadata
            metadata['is_recording'] = event_saver.is_recording
            if event_saver.is_recording:
                metadata['recording_classes'] = list(sorted(event_saver.detected_classes))
                metadata['recording_frame_count'] = event_saver.frame_count
            
            # Send frame and metadata
            await websocket.send_json({
                "type": "frame",
                "image": jpg_as_text,
                "metadata": metadata
            })
            
            # Control frame rate
            await asyncio.sleep(1 / Config.TARGET_FPS)
            
    except WebSocketDisconnect:
        print(f"✗ WebSocket disconnected (Remaining: {len(active_connections) - 1})")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
    finally:
        # Remove from active connections
        active_connections.discard(websocket)
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level="info",
        ws_ping_interval=Config.WEBSOCKET_PING_INTERVAL,
        ws_ping_timeout=Config.WEBSOCKET_PING_TIMEOUT
    )
