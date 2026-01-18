import numpy as np
from datetime import timedelta

class OccupancyManager:
    def __init__(self, zone_names=None):
        # FIX: Accept zone_names argument to match main.py call
        if zone_names is None:
            self.zone_names = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
        else:
            self.zone_names = zone_names
            
        # Paper 5.1: "Space 5 and one side room (Space 3) to be the main office rooms"
        self.main_offices = ["SPACE5-1", "SPACE3-1"]
        
        # Paper 5.1: "The other rooms are used as conference rooms"
        self.conference_rooms = [z for z in self.zone_names if z not in self.main_offices]

    def get_occupancy_state(self, zone_name, current_dt):
        """
        Returns the state vector features required by the Agent:
        - Current relative occupancy
        - Forecast 1 hour ahead
        - Forecast 2 hours ahead
        """
        # 1. Current
        curr = self._get_occ_at_time(zone_name, current_dt)
        
        # 2. Next 1h
        dt_1h = current_dt + timedelta(hours=1)
        nxt_1 = self._get_occ_at_time(zone_name, dt_1h)
        
        # 3. Next 2h
        dt_2h = current_dt + timedelta(hours=2)
        nxt_2 = self._get_occ_at_time(zone_name, dt_2h)
        
        return {'current': curr, 'next_1h': nxt_1, 'next_2h': nxt_2}

    def get_next_occupancy(self, current_dt):
        """Returns dictionary of {zone: occ_value} for EnergyPlus simulation"""
        occ_map = {}
        for z in self.zone_names:
            occ_map[z] = self._get_occ_at_time(z, current_dt)
        return occ_map

    def _get_occ_at_time(self, zone, dt):
        """Calculates deterministic occupancy based on Paper rules"""
        hour = dt.hour
        weekday = dt.weekday() # 0=Mon, 6=Sun

        # Weekend = Empty (Paper: "except holidays without presence")
        if weekday >= 5:
            return 0.0

        # Main Offices (Fixed Schedule 8:00 am - 4:00 pm)
        if zone in self.main_offices:
            if 8 <= hour < 16:
                return 1.0 # Full capacity
            else:
                return 0.0

        # Conference Rooms (Changing occupancy)
        # We use a deterministic hash so the "randomness" is consistent across episodes
        if zone in self.conference_rooms:
            # Simple pseudo-random schedule based on day/hour
            # This ensures "changing occupancy" as per paper, but reproducible training
            seed = weekday * 24 + hour + hash(zone)
            np.random.seed(seed % (2**32 - 1))
            
            # Conference happens randomly between 9am and 5pm
            if 9 <= hour < 17:
                # 30% chance a conference is happening this hour
                return 1.0 if np.random.random() < 0.3 else 0.0
            return 0.0
            
        return 0.0

    def calculate_complaint(self, zone, current_temp, occupancy):
        """
        Paper 4.3 (Eq 7 & 8):
        - If unoccupied: 0
        - Target is occupants mean comfort (assume 21.0 C)
        - If |T - Target| <= 1.0: 0 (Tolerance)
        - Else: Magnitude of difference
        """
        if occupancy <= 0.01:
            return 0.0
            
        target_temp = 21.0 
        diff = abs(current_temp - target_temp)
        
        if diff <= 1.0:
            return 0.0
        
        return diff