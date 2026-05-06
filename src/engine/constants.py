from dataclasses import dataclass
from enum import Enum

@dataclass
class StressRange(Enum):
    residential = 0.3
    collector_road = 0.4
    arterial_road = 0.5
    highway = 1.0
    interstate = 1.2
    freeway = 1.5
    
    
@dataclass
class DamageType(Enum):
    CRACK = "crack"
    POTHOLE = "pothole"    
    
@dataclass
class CrackSubtype(Enum):
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"
    ALLIGATOR = "alligator"
    
@dataclass
class PotholeSubtype(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    
@dataclass
class UnitTypes(Enum):
    cm = "cm"
    inch = "inch"