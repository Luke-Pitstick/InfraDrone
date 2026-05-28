from .constants import UnitTypes


class BaseEngine:
    """Shared configuration for detection and segmentation engines."""

    def __init__(self, unit_type: UnitTypes, width_ratio: float, height_ratio: float):
        """Store unit conversion settings used when reporting measurements.

        Args:
            unit_type: Target unit for measurements (cm, inch, or px).
            width_ratio: Scale factor applied to horizontal pixel distances.
            height_ratio: Scale factor applied to vertical pixel distances.
        """
        self.unit_type = unit_type
        self.px_to_unit_ratio = {
            UnitTypes.cm: 1,
            UnitTypes.inch: 2.54,
            UnitTypes.px: 1,
        }[unit_type]
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
