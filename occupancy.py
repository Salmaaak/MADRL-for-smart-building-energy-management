import numpy as np
from datetime import timedelta

class OccupancyManager:
    def __init__(self):
        # Zones as per paper
        self.zones = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
        self.max_capacity = {z: 5 for z in self.zones} # Default
        self.max_capacity["SPACE5-1"] = 20 # Center
        self.max_capacity["SPACE3-1"] = 11 # Large side room
        self.max_capacity["SPACE1-1"] = 11

        # Comfort preference (Can be randomized per episode)
        self.comfort_temp = {z: 21.0 for z in self.zones} 

    def get_occupancy_state(self, zone, current_dt):
        """Returns normalized occupancy for t, t+1h, t+2h"""
        curr = self._sample_occ(zone, current_dt)
        next_1 = self._sample_occ(zone, current_dt + timedelta(hours=1))
        next_2 = self._sample_occ(zone, current_dt + timedelta(hours=2))
        
        return {
            'current': curr,
            'next_1h': next_1,
            'next_2h': next_2
        }

    def get_next_occupancy(self, current_dt):
        """Returns dict of occupancy values for simulation actuators"""
        return {z: self._sample_occ(z, current_dt) for z in self.zones}

    def calculate_complaint(self, zone, current_temp, current_occ_pct):
        # Eq 7 & 8 in Paper
        # Complaint only if people are present
        if current_occ_pct <= 0.01:
            return 0.0
        
        desired = self.comfort_temp[zone]
        diff = abs(current_temp - desired)
        
        # Deadband of 1.0 degree
        if diff <= 1.0:
            return 0.0
        return diff

    def _sample_occ(self, zone, dt):
        # Simplified Stochastic Logic based on Paper Section 5.1
        # Space 5 & 3: 8am-4pm
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        
        if is_weekend:
            return 0.0
        
        base_occ = 0.0
        if 8 <= hour < 16:
            base_occ = 1.0
        
        # Add stochastic noise or variability here if strictly following stochastic nature
        return base_occ