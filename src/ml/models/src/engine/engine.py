"""
This module contains the engine for the machine learning model.
"""

from dataclasses import dataclass
from enum import Enum

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from PIL import Image

from constants import StressRange, DamageType, CrackSubtype, PotholeSubtype, UnitTypes


@dataclass
class Measurement:
    def __init__(self, value: float, unit: UnitTypes):
        self.value: float
        self.unit: UnitTypes
        
    def to_centimeters(self, value: float) -> float:
        if self.unit == UnitTypes.cm:
            return value
        elif self.unit == UnitTypes.inch:
            return value * 2.54
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
        
    def to_inches(self, value: float) -> float:
        if self.unit == UnitTypes.cm:
            return value / 2.54
        elif self.unit == UnitTypes.inch:
            return value
        else:
            raise ValueError(f"Invalid measurement unit: {self}")
   
@dataclass  
class Dimensions:
    width: Measurement
    length: Measurement
    
    
@dataclass
class Damage:
    type: DamageType
    severity: int
    confidence: float
    dimensions: Dimensions
    subtype: CrackSubtype | PotholeSubtype
    stress_range: StressRange
    num_connections: int

    
class Engine:
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass