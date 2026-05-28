"""Enumerations shared across InfraDrone engine outputs and measurements."""

from enum import Enum


class StressRange(float, Enum):
    """Road-class stress multipliers used when assessing pavement damage.

    Higher values represent higher-traffic road classes where equivalent damage
    may imply greater structural or operational concern.
    """

    RESIDENTIAL = 0.3
    COLLECTOR_ROAD = 0.4
    ARTERIAL_ROAD = 0.5
    HIGHWAY = 1.0
    INTERSTATE = 1.2
    FREEWAY = 1.5


class DamageType(str, Enum):
    """Top-level road-damage categories detected by the engine."""

    CRACK = "crack"
    POTHOLE = "pothole"


class CrackSubtype(str, Enum):
    """Crack subtype labels assigned during segmentation post-processing."""

    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"
    ALLIGATOR = "alligator"
    UNKNOWN = "unknown"


class PotholeSubtype(str, Enum):
    """Size-based pothole subtype labels."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class UnitTypes(str, Enum):
    """Supported units for engine measurements."""

    cm = "cm"
    inch = "inch"
    px = "px"
    deg = "deg"
    rad = "rad"
