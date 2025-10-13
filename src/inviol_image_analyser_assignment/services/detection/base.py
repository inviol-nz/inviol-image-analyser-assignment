"""Abstract interface for detection backends.

Each backend exposes a uniform API so the rest of the service (rules, API)
doesn't care whether detections come from YOLO, a hosted API, etc.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from inviol_image_analyser_assignment.models import Detection

class DetectionBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def predict(self, image_bytes: bytes) -> List[Detection]: ...
