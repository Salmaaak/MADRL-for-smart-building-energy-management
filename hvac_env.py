import gymnasium as gym
import numpy as np
import threading
import queue
import sys
import time
import shutil
from datetime import datetime, timedelta

# --- POINT TO YOUR ENERGYPLUS INSTALLATION ---
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

        # --- Paper 4.1: State Space (Size 10) ---
        self.observation_space = gym.spaces.Box(low=-50, high=2000, shape=(10,), dtype=np.float32)
        
        # --- Paper 4.2: Action Space (15C to 25C) ---
        self.action_space = gym.spaces.Discrete(11)
        self.action_map = np.linspace(15.0, 25.0, 11)

        self.api = EnergyPlusAPI()
        self.ep_state = None
        self.obs_queue = queue.Queue(maxsize=1)
        self.act_queue = queue.Queue(maxsize=1)
        self.sim_thread = None
        self.sim_running = False
        self.zone_names = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
        self.occupancy_manager = None
        self.episode_idx = 0
        
        # Reward Hyperparameters
        self.lambda_e = 0.008
        self.lambda_m = 0.12
        self.l_clip = 5.0 

    def reset(self, seed=None, options=None):
        # 1. Stop existing simulation
        self.stop_simulation()
        
        # 2. Clear any stale data in queues
        with self.obs_queue.mutex:
            self.obs_queue.queue.clear()
        with self.act_queue.mutex:
            self.act_queue.queue.clear()

        self.sim_running = True
        self.episode_idx += 1
        self.handles = {} 
        
        # 3. Start Thread
        self.sim_thread = threading.Thread(target=self._run_energyplus)
        self.sim_thread.start()

        # 4. Wait for first observation safely
        try:
            val = self.obs_queue.get(timeout=30)
        except queue.Empty:
            self.stop_simulation()
            raise RuntimeError("EnergyPlus failed to start (Timeout).")
            
        # 5. Handle Crash (NoneType error fix)
        if val is None:
            self.stop_simulation()
            raise RuntimeError(f"EnergyPlus crashed immediately on Episode {self.episode_idx}. Check {self.output_dir} for .err logs.")

        obs_dict, info = val
        return obs_dict, info

    def step(self, actions):
        total_energy = 0
        total_complaints = 0
        obs_dict = {}
        
        # Paper: 1 RL step = 1 hour = 4 x 15min sim steps
        for _ in range(4):
            if not self.sim_running: break
            
            self.act_queue.put(actions)
            
            try:
                result = self.obs_queue.get(timeout=10)
                if result is None: 
                    self.sim_running = False
                    break
                
                obs_dict, info = result
                total_energy += info['energy']       
                total_complaints += info['complaint'] 
                
            except queue.Empty:
                self.sim_running = False
                break

        # Reward Calculation
        e_kwh = total_energy / 3600000.0
        if e_kwh > self.l_clip:
            e_flat = (e_kwh - self.l_clip) * 0.1 + self.l_clip
        else:
            e_flat = e_kwh
            
        reward = - (self.lambda_e * e_flat + self.lambda_m * total_complaints)
        
        done = not self.sim_running
        return obs_dict, reward, done, False, {}

    def stop_simulation(self):
        self.sim_running = False
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=3)
        self.ep_state = None

    def _run_energyplus(self):
        self.ep_state = self.api.state_manager.new_state()
        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(self.ep_state, self._callback_step)
        self.api.runtime.set_console_output_status(self.ep_state, False)
        
        # FIX: Unique timestamp per episode avoids "Permission Denied" on Windows file locks
        timestamp = int(time.time())
        run_dir = f"{self.output_dir}/ep_out_{self.episode_idx}_{timestamp}"
        
        args = [
            '-d', run_dir, 
            '-w', self.epw_path, 
            '-r', 
            self.idf_path
        ]
        
        # Capture exit code
        exit_code = self.api.runtime.run_energyplus(self.ep_state, args)
        
        if exit_code != 0:
            print(f"\n[Error] EnergyPlus Simulation ended with exit code: {exit_code}")
        
        self.obs_queue.put(None) 

    def _callback_step(self, state_arg):
        if not self.sim_running: 
            return

        if not self.handles: 
            if not self._init_handles(state_arg): return

        # -- GET DATA --
        current_dt, time_feats, weather_feats = self._get_global_features(state_arg)
        
        # Occupancy & Complaints
        next_occ = self.occupancy_manager.get_next_occupancy(current_dt)
        self._apply_occupancy(state_arg, next_occ)
        
        obs_dict = {}
        step_energy = 0
        step_complaint = 0

        for zone in self.zone_names:
            z_temp = self.api.exchange.get_variable_value(state_arg, self.handles['zones'][zone]['temp'])
            z_joules = self.api.exchange.get_variable_value(state_arg, self.handles['zones'][zone]['power']) * 900.0
            step_energy += z_joules

            occ_data = self.occupancy_manager.get_occupancy_state(zone, current_dt)
            complaint = self.occupancy_manager.calculate_complaint(zone, z_temp, occ_data['current'])
            step_complaint += complaint

            z_feats = np.array([z_temp, occ_data['current'], occ_data['next_1h'], occ_data['next_2h']])
            obs_dict[zone] = np.concatenate([time_feats, weather_feats, z_feats], dtype=np.float32)

        self.obs_queue.put((obs_dict, {'energy': step_energy, 'complaint': step_complaint}))

        # -- GET ACTION --
        try:
            # Wait for action from Agent
            actions = self.act_queue.get(timeout=2)
            self._apply_actions(state_arg, actions)
        except queue.Empty:
            # If agent is slow, default to keeping current setpoints (or do nothing)
            pass

    def _init_handles(self, state):
        if not self.api.exchange.api_data_fully_ready(state): return False
        self.handles = {'actuators': {}, 'cool_act': {}, 'schedules': {}, 'zones': {}, 'weather': {}}
        
        self.handles['weather']['temp'] = self.api.exchange.get_variable_handle(state, "Site Outdoor Air Drybulb Temperature", "Environment")
        
        for zone in self.zone_names:
            self.handles['actuators'][zone] = self.api.exchange.get_actuator_handle(state, "Zone Temperature Control", "Heating Setpoint", zone)
            self.handles['cool_act'][zone] = self.api.exchange.get_actuator_handle(state, "Zone Temperature Control", "Cooling Setpoint", zone)
            self.handles['schedules'][zone] = self.api.exchange.get_actuator_handle(state, "Schedule:Constant", "Schedule Value", f"OCC-SCHEDULE-{zone}")
            self.handles['zones'][zone] = {
                'temp': self.api.exchange.get_variable_handle(state, "Zone Air Temperature", zone),
                'power': self.api.exchange.get_variable_handle(state, "Zone Air System Sensible Heating Rate", zone)
            }
        return True

    def _get_global_features(self, state):
        year = self.api.exchange.year(state)
        month = self.api.exchange.month(state)
        day = self.api.exchange.day_of_month(state)
        hour = self.api.exchange.hour(state)
        minute = self.api.exchange.minutes(state)
        
        # TIMEDELTA FIX for "Hour 24" or "Minute 60"
        dt_base = datetime(year, month, day)
        current_dt = dt_base + timedelta(hours=hour, minutes=minute)
        
        day_of_week = float(current_dt.weekday()) # 0-6
        min_of_day = float(current_dt.hour * 60 + current_dt.minute)
        cal_week = float(current_dt.isocalendar()[1])

        out_temp = self.api.exchange.get_variable_value(state, self.handles['weather']['temp'])
        
        # Solar Placeholders (Paper requires 10 inputs)
        direct_solar = 0.0 
        indirect_solar = 0.0 

        time_vec = np.array([day_of_week, min_of_day, cal_week])
        weather_vec = np.array([out_temp, direct_solar, indirect_solar])
        
        return current_dt, time_vec, weather_vec

    def _apply_occupancy(self, state, occ_map):
        for z, val in occ_map.items():
            if self.handles['schedules'].get(z, -1) > -1:
                self.api.exchange.set_actuator_value(state, self.handles['schedules'][z], val)

    def _apply_actions(self, state, actions):
        for z, idx in actions.items():
            setpoint = self.action_map[idx]
            if self.handles['actuators'].get(z, -1) > -1:
                self.api.exchange.set_actuator_value(state, self.handles['actuators'][z], setpoint)
                # Safety Gap
                self.api.exchange.set_actuator_value(state, self.handles['cool_act'][z], 30.0)