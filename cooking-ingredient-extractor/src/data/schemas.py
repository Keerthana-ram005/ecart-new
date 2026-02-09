"""
Shared data schemas to prevent conflicts.
All team members MUST use these schemas.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Ingredient:
    """Ingredient data structure"""
    name: str
    confidence: float
    quantity: Optional[str] = None
    unit: Optional[str] = None
    source: str = "unknown"  # "visual", "audio", "text"
    timestamp: Optional[float] = None  # Video timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionResult:
    """Base class for detection results"""
    module: str  # "visual", "audio", "text"
    ingredients: List[Ingredient]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VideoInfo:
    """Video metadata"""
    path: str
    duration: float
    fps: float
    resolution: tuple
    frame_count: int

@dataclass
class FinalResult:
    """Final fused result"""
    video_info: VideoInfo
    ingredients: List[Ingredient]
    processing_stats: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            "video": self.video_info.__dict__,
            "ingredients": [ing.__dict__ for ing in self.ingredients],
            "stats": self.processing_stats,
            "timestamp": self.timestamp
        }
    
    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)