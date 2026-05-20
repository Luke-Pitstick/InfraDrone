from enum import Enum


class StressRange(float, Enum):
    RESIDENTIAL = 0.3
    COLLECTOR_ROAD = 0.4
    ARTERIAL_ROAD = 0.5
    HIGHWAY = 1.0
    INTERSTATE = 1.2
    FREEWAY = 1.5


class DamageType(str, Enum):
    CRACK = "crack"
    POTHOLE = "pothole"


class CrackSubtype(str, Enum):
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"
    ALLIGATOR = "alligator"


class PotholeSubtype(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class UnitTypes(str, Enum):
    cm = "cm"
    inch = "inch"
    px = "px"
