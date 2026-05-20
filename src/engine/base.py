from .constants import UnitTypes

class BaseEngine:
    def __init__(self, unit_type: UnitTypes, width_ratio: float, height_ratio: float):
        self.unit_type = unit_type
        self.px_to_unit_ratio = {
            UnitTypes.cm: 1,
            UnitTypes.inch: 2.54,
            UnitTypes.px: 1,
        }[unit_type]
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        