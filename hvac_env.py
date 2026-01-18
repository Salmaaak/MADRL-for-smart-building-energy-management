import gymnasium as gym
import numpy as np
import threading
import queue
import sys
from datetime import datetime, timedelta

# --- CRITICAL: Point to your EnergyPlus ROOT folder ---
energyplus_install_path = r'C:\EnergyPlus-24.1.0' 
if energyplus_install_path not in sys.path:
    sys.path.insert(0, energyplus_install_path)

from pyenergyplus.api import EnergyPlusAPI

class EnergyPlusEnv(gym.Env):
    def __init__(self, idf_path, epw_path, output_dir="results"):
        super().__init__()
        self.idf_path = idf_path
        self.epw_path = epw_path
        self.output_dir = output_dir

        self.observation_space = gym.spaces.Box(low=-50, high=2000, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(11)
        self.action_map = np.linspace(15.0, 25.0, 11)

        self.api = EnergyPlusAPI()
        self.ep_state = None
        self.obs_queue = queue.Queue(maxsize=1)
        self.act_queue = queue.Queue(maxsize=1)
        self.sim_thread = None
        self.sim_running = False
        self.episode_idx = 0

        self.handles = {}
        self.zone_names = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
        self.occupancy_manager = None 

    def reset(self, seed=None, options=None):
        self.stop_simulation()
        self.sim_running = True
        self.episode_idx += 1
        
        self.sim_thread = threading.Thread(target=self._run_energyplus)
        self.sim_thread.start()

        try:
            result = self.obs_queue.get(timeout=20)
        except queue.Empty:
            raise TimeoutError("EnergyPlus failed to start.")
            
        if result is None:
            raise RuntimeError("Simulation stopped unexpectedly during reset.")

        obs_dict, _, _, info = result
        return obs_dict, info

    def step(self, actions):
        if not self.sim_running:
            return {}, 0, True, False, {}

        self.act_queue.put(actions)

        try:
            result = self.obs_queue.get(timeout=10)
        except queue.Empty:
            self.sim_running = False
            return {}, 0, True, False, {}

        if result is None: 
            self.sim_running = False
            return {}, 0, True, False, {}

        obs_dict, done, truncated, info = result
        reward = self._calculate_reward(info)
        return obs_dict, reward, done, truncated, info

    def _calculate_reward(self, info):
        lambda_e = 0.008
        lambda_m = 0.12 
        energy_kwh = info.get('energy', 0) / 3600000.0
        complaints = info.get('complaint', 0)
        return -(lambda_e * energy_kwh + lambda_m * complaints)

    def stop_simulation(self):
        if self.sim_running:
            self.sim_running = False
            if self.sim_thread:
                self.sim_thread.join(timeout=2)

    def _run_energyplus(self):
        self.ep_state = self.api.state_manager.new_state()
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            self.ep_state, self._callback_step
        )
        self.api.runtime.set_console_output_status(self.ep_state, False)
        args = ['-d', f"{self.output_dir}/ep_out_{self.episode_idx}", '-w', self.epw_path, self.idf_path]
        self.api.runtime.run_energyplus(self.ep_state, args)
        self.obs_queue.put(None)

    def _callback_step(self, state_arg):
        if not self.sim_running: return

        if not self.handles:
            if not self._init_handles(state_arg): return

        # FIX 2: Robust Time Handling
        current_dt, time_feats, weather_feats = self._get_global_features(state_arg)

        next_occ_map = self.occupancy_manager.get_next_occupancy(current_dt)
        self._apply_occupancy_schedules(state_arg, next_occ_map)

        obs_dict = {}
        total_energy = 0.0
        total_complaints = 0.0

        for zone in self.zone_names:
            z_temp = self.api.exchange.get_variable_value(state_arg, self.handles['zones'][zone]['temp'])
            z_energy = self.api.exchange.get_variable_value(state_arg, self.handles['zones'][zone]['energy'])
            total_energy += z_energy

            occ_data = self.occupancy_manager.get_occupancy_state(zone, current_dt)
            curr_occ = occ_data['current']
            complaint = self.occupancy_manager.calculate_complaint(zone, z_temp, curr_occ)
            total_complaints += complaint

            zone_features = np.array([z_temp, curr_occ, occ_data['next_1h'], occ_data['next_2h']], dtype=np.float32)
            
            # --- FIX: Correct variable name used here (weather_feats) ---
            obs_dict[zone] = np.concatenate([time_feats, weather_feats, zone_features])

        info = {'energy': total_energy, 'complaint': total_complaints, 'time': current_dt}
        self.obs_queue.put((obs_dict, False, False, info))

        try:
            actions = self.act_queue.get(timeout=10)
        except queue.Empty:
            return

        self._apply_actions(state_arg, actions)

    def _init_handles(self, state):
        if not self.api.exchange.api_data_fully_ready(state): return False
        self.handles['actuators'] = {}
        self.handles['schedules'] = {}
        self.handles['zones'] = {}
        self.handles['weather'] = {
            'temp': self.api.exchange.get_variable_handle(state, "Site Outdoor Air Drybulb Temperature", "Environment"),
            'direct_solar': self.api.exchange.get_variable_handle(state, "Site Direct Solar Radiation Rate per Area", "Environment"),
            'diffuse_solar': self.api.exchange.get_variable_handle(state, "Site Diffuse Solar Radiation Rate per Area", "Environment"),
        }
        for zone in self.zone_names:
            self.handles['actuators'][zone] = self.api.exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint", zone)
            self.handles['schedules'][zone] = self.api.exchange.get_actuator_handle(state, "Schedule:Constant", "Schedule Value", f"OCC-SCHEDULE-{zone}")
            self.handles['zones'][zone] = {
                'temp': self.api.exchange.get_variable_handle(state, "Zone Air Temperature", zone),
                'energy': self.api.exchange.get_variable_handle(state, "Zone Air System Sensible Heating Energy", zone)
            }
        return True

    def _get_global_features(self, state):
        year = self.api.exchange.year(state)
        month = self.api.exchange.month(state)
        day = self.api.exchange.day_of_month(state)
        hour = self.api.exchange.hour(state)
        minute = self.api.exchange.minutes(state)

        # FIX 3: Timedelta for robust time handling
        dt_base = datetime(year, month, day)
        current_dt = dt_base + timedelta(hours=hour, minutes=minute)
        
        dt_hour = current_dt.hour
        dt_minute = current_dt.minute

        min_of_day = dt_hour * 60 + dt_minute
        day_of_week = current_dt.weekday()
        cal_week = current_dt.isocalendar()[1]
        
        time_feats = np.array([min_of_day, day_of_week, cal_week], dtype=np.float32)
        out_temp = self.api.exchange.get_variable_value(state, self.handles['weather']['temp'])
        dir_solar = self.api.exchange.get_variable_value(state, self.handles['weather']['direct_solar'])
        dif_solar = self.api.exchange.get_variable_value(state, self.handles['weather']['diffuse_solar'])
        weather_feats = np.array([out_temp, dir_solar, dif_solar], dtype=np.float32)

        return current_dt, time_feats, weather_feats

    def _apply_occupancy_schedules(self, state, occ_map):
        for zone, value in occ_map.items():
            handle = self.handles['schedules'].get(zone, -1)
            if handle > -1: self.api.exchange.set_actuator_value(state, handle, value)

    def _apply_actions(self, state, actions):
        for zone, act_idx in actions.items():
            setpoint = self.action_map[act_idx]
            handle = self.handles['actuators'].get(zone, -1)
            if handle > -1: self.api.exchange.set_actuator_value(state, handle, setpoint)