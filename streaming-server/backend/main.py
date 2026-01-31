"""
FastAPI WebSocket Video Streaming Server
Streams video with YOLO object detection to web clients
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import base64
import asyncio
import json
from pathlib import Path
from typing import Set
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from backend.tracker import ObjectTracker, CameraCapture

# Initialize FastAPI app
app = FastAPI(title="Object Tracking Stream", version="1.0.0")

# Global tracker and camera
tracker: ObjectTracker = None
camera: CameraCapture = None
active_connections: Set[WebSocket] = set()

# Mount static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize tracker and camera on startup"""
    global tracker, camera
    
    print("\n" + "="*60)
    print("🚀 Starting Object Tracking Streaming Server")
    print("="*60)
    
    try:
        # Initialize tracker
        tracker = ObjectTracker()
        
        # Initialize camera
        camera = CameraCapture()
        
        print("\n✓ Server initialization complete")
        print(f"✓ Configuration: {Config.get_info()}")
        print(f"\n📡 Server running at http://{Config.HOST}:{Config.PORT}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global camera, active_connections
    
    print("\n🛑 Shutting down server...")
    
    # Close all active connections
    for connection in active_connections.copy():
        try:
            await connection.close()
        except:
            pass
    active_connections.clear()
    
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
            
            # Encode frame to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
            success, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            
            if not success:
                continue
            
            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
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
