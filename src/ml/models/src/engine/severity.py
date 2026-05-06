import math
from enum import Enum
from engine.engine import Damage, Measurement, StressRange



class Severity():
    def __init__(self, damage: Damage):
        self.severity: float = 0
        self.damage : Damage = damage
        
        self.width: Measurement = damage.dimensions.width
        self.length: Measurement = damage.dimensions.length
        
        # material constants
        self.C : float = 1.0
        self.m : float = 1.0
        
        # geometry factor
        self.Y : float = 1.1
        
        # stress range
        self.stress_range: StressRange = damage.stress_range.value
        
        # sigmoid parameters
        self.growth_rate : float = 0.15
        
    def calculate_stress_intensity_factor(self) -> float:
        """
        Calculate the stress intensity factor.
        
        K = Y * delta(sigma) * sqrt(pi * a)
        where:
        - Y = geometry factor
        - delta(sigma) = stress range
        - a = crack length
        """
        
        return self.Y * self.stress_range * math.sqrt(math.pi * self.length.value)
        
    def _calculate_crack_growth_rate(self) -> float:
        """
        Calculate the crack growth rate.
        
        da/dN = C(delta(K))**m
        where:
        - a = crack length
        - N = number of load cycles
        - C, m = material constants
        - delta(K) = the stress intensity factor range = Y * delta(sigma) * sqrt(pi * a)
          where: 
            Y = geometry factor
            delta(sigma) = stress range
        """
        delta_K = self.calculate_stress_intensity_factor()
        deriv = self.C * (delta_K**self.m)
        return deriv
        
    def _calculate_N_cycles_for_damage_growth(self, ending_length: float) -> float:
        """
        Calculate the number of load cycles for the damage to grow to the next size.
        
        N = (start_len**1-m/2 - ending_len**1-m/2) / (C * (delta(K))**m) * (1 - m/2)
        where:
        - N = number of load cycles
        - C, m = material constants
        - delta(K) = the stress intensity factor range
        """
        delta_K = self.calculate_stress_intensity_factor()
        numerator = (self.length.value**(1-self.m/2) - ending_length**(1-self.m/2))
        denominator = self.C * (delta_K**self.m) * (1 - self.m/2)
        return numerator / denominator
    
    def _normalize_severity(self, raw_severity: float) -> float:
        k = -math.log(0.1) / self.growth_rate
        return 1.0 - math.exp(-k * raw_severity)
        
        
    def calculate_severity(self, cycle_count: int = 100000) -> float:
        """
        Calculate the severity of the damage. Allows users to specify the number of load cycles for the damage to grow to the next size.
        
        Severity = crack growth rate * cycle_count * number of connections * (crack width / 5)
        where:
        - crack growth rate = da/dN
        - cycle_count = number of load cycles
        - number of connections = number of connections in the damage
        - crack width = crack width
        - 5 = arbitrary number to lower weight of width
        """
        crack_growth_rate = self._calculate_crack_growth_rate()
        
        predicted_growth = crack_growth_rate * cycle_count
        
        raw_severity = (
            predicted_growth 
            * max(1, self.damage.num_connections) 
            * (self.width.value / 5)
        )
        
        raw_severity = max(0.0, raw_severity)
        
        # Normalize to 0-1 using sigmoid function
        severity = self._normalize_severity(raw_severity)
        return severity
        
        
        
    