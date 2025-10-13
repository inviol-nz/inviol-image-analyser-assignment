from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from inviol_image_analyser_assignment.models import Detection

class DetectionBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def predict(self, image_bytes: bytes) -> List[Detection]: ...
