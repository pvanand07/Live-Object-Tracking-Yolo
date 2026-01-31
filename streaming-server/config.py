"""
Configuration settings for the streaming server
"""

import os

class Config:
    """Server configuration"""
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Camera settings
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
    CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", 1280))
    CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", 720))
    TARGET_FPS = int(os.getenv("TARGET_FPS", 30))
    
    # YOLO settings
    YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
    TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
    
    # Streaming settings
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 80))
    MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", 10))
    
    # WebSocket settings
    WEBSOCKET_PING_INTERVAL = int(os.getenv("WEBSOCKET_PING_INTERVAL", 20))
    WEBSOCKET_PING_TIMEOUT = int(os.getenv("WEBSOCKET_PING_TIMEOUT", 20))
    
    @classmethod
    def get_info(cls):
        """Get configuration info as dict"""
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "camera_resolution": f"{cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}",
            "target_fps": cls.TARGET_FPS,
            "yolo_model": cls.YOLO_MODEL,
            "jpeg_quality": cls.JPEG_QUALITY,
        }
