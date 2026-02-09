# src/models/visual_detector/detector.py
from dataclasses import dataclass
from typing import List
import cv2
import torch
from ..data.schemas import Ingredient, DetectionResult

@dataclass
class DetectionConfig:
    model_path: str = "yolov5s.pt"
    confidence_threshold: float = 0.5
    frame_interval: int = 30  # Process every 30th frame

class VisualDetector:
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                        path=self.config.model_path)
            print("✅ Visual model loaded")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    
    def process_video(self, video_path: str) -> DetectionResult:
        """Process video and extract ingredients"""
        import time
        start_time = time.time()
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Detect ingredients
        ingredients = []
        for frame in frames:
            detections = self.detect_objects(frame)
            ingredients.extend(detections)
        
        # Create result
        result = DetectionResult(
            module="visual",
            ingredients=ingredients,
            processing_time=time.time() - start_time,
            metadata={"frames_processed": len(frames)}
        )
        
        return result
    
    def extract_frames(self, video_path: str):
        """Extract frames at intervals"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.config.frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_objects(self, frame):
        """Detect objects in single frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model(frame_rgb)
        
        # Parse results
        ingredients = []
        for *box, conf, cls in results.xyxy[0]:
            if conf >= self.config.confidence_threshold:
                label = results.names[int(cls)]
                if self.is_food_item(label):
                    ingredient = Ingredient(
                        name=label,
                        confidence=float(conf),
                        source="visual",
                        metadata={"bbox": box}
                    )
                    ingredients.append(ingredient)
        
        return ingredients
    
    def is_food_item(self, label: str) -> bool:
        """Check if label is food-related"""
        food_keywords = [
            'apple', 'banana', 'orange', 'tomato', 'onion', 'carrot',
            'bowl', 'bottle', 'cup', 'knife', 'spoon'
        ]
        return any(keyword in label.lower() for keyword in food_keywords)