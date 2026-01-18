import sys
import os
import csv
import numpy as np
import msvcrt 

# Update this path to where your EnergyPlus is installed
sys.path.insert(0, 'C:\\EnergyPlus-24.1.0') 

from hvac_env import EnergyPlusEnv
from occupancy import OccupancyManager
from agent import SharedAgent

# Configuration
IDF_FILE = "5ZoneAirCooled.idf"
EPW_FILE = "MAR_RK_Rabat-Sale.AP.601350_TMYx.2009-2023.epw"
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_model.pth")

def setup_dirs():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

def init_csv_files(resume=False):
    # Step Metrics Header
    step_file = f"{LOG_DIR}/step_metrics.csv"
    step_header = ["Episode", "Step", "Reward", "Energy_J", "Complaints", 
                   "Mean_Zone_Temp", "Max_Occupancy", "Day", "Hour", "Minute", "Timestamp"]
    
    # Only write header if file doesn't exist or we are NOT resuming
    if not resume or not os.path.exists(step_file):
        with open(step_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(step_header)

    # Episode Metrics Header
    ep_file = f"{LOG_DIR}/episode_metrics.csv"
    ep_header = ["Episode", "Total_Reward", "Total_Energy_J", "Total_Complaints", "Avg_Reward"]
    
    if not resume or not os.path.exists(ep_file):
        with open(ep_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ep_header)

def log_step(episode, step, reward, info, obs_dict):
    temps = [state[6] for state in obs_dict.values()]
    avg_temp = np.mean(temps) if temps else 0
    
    occs = [state[7] for state in obs_dict.values()]
    max_occ = np.max(occs) if occs else 0
    
    # Extract Time Details
    timestamp = info.get('time', 'N/A')
    day = "N/A"
    hour = "N/A"
    minute = "N/A"
    
    if hasattr(timestamp, 'weekday'): # Check if it's a valid datetime object
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day = day_names[timestamp.weekday()]
        hour = timestamp.hour
        minute = timestamp.minute

    with open(f"{LOG_DIR}/step_metrics.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, step, reward, info.get('energy', 0), 
                         info.get('complaint', 0), avg_temp, max_occ, 
                         day, hour, minute, timestamp])

def log_episode(episode, total_reward, total_energy, total_complaints, steps):
    avg_reward = total_reward / steps if steps > 0 else 0
    with open(f"{LOG_DIR}/episode_metrics.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode, total_reward, total_energy, total_complaints, avg_reward])

def run_training():
    setup_dirs()
    
    env = EnergyPlusEnv(IDF_FILE, EPW_FILE)
    occ_manager = OccupancyManager()
    env.occupancy_manager = occ_manager
# Change input_dim from 10 to 8
    agent = SharedAgent(input_dim=8, action_dim=11)
    # --- AUTO-RESUME LOGIC ---
    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f">>> Found checkpoint: {CHECKPOINT_PATH}")
        last_ep = agent.load_checkpoint(CHECKPOINT_PATH)
        start_episode = last_ep
        print(f">>> Resuming from Episode {start_episode + 1}")
        init_csv_files(resume=True)
    else:
        print(">>> No checkpoint found. Starting fresh.")
        init_csv_files(resume=False)

    num_episodes = 1000
    stop_training = False 
    
    try:
        for episode in range(start_episode, num_episodes):
            if stop_training: break

            print(f"--- Starting Episode {episode + 1} (Press 'q' to stop) ---")
            
            try:
                obs_dict, info = env.reset()
            except Exception as e:
                print(f"Error starting episode: {e}")
                # If reset fails, we try to save progress so we don't lose the episode count
                agent.save_checkpoint(CHECKPOINT_PATH, episode)
                break

            done = False
            total_reward = 0
            total_energy = 0
            total_complaints = 0
            step_count = 0
            
            while not done:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key.lower() == b'q':
                        print("\n>>> Stop signal received (q). Stopping after this episode...")
                        stop_training = True

                actions = agent.select_actions(obs_dict)
                next_obs_dict, reward, done, truncated, info = env.step(actions)
                
                if done:
                    next_obs_dict = obs_dict
                    if not info: 
                        print("Simulation ended or crashed.")
                        break
                
                agent.store_transition(obs_dict, actions, reward, next_obs_dict, done)
                agent.train()
                
                if info and 'energy' in info:
                    total_reward += reward
                    total_energy += info['energy']
                    total_complaints += info['complaint']
                    step_count += 1
                    
                    log_step(episode + 1, step_count, reward, info, obs_dict)
                    
                    if step_count % 100 == 0:
                        e = info.get('energy', 0)
                        c = info.get('complaint', 0)
                        print(f"Step {step_count}: Reward {reward:.4f}, Energy {e:.2f}J, Complaints {c:.2f}")

                obs_dict = next_obs_dict
                if stop_training: break

            agent.update_target_network()
            
            # --- SAVE CHECKPOINT EVERY EPISODE ---
            # This ensures if it crashes at Ep 64, next time it starts at 64 (or 65 if completed)
            agent.save_checkpoint(CHECKPOINT_PATH, episode + 1)
            
            log_episode(episode + 1, total_reward, total_energy, total_complaints, step_count)
            print(f"Episode {episode + 1} End. Total Reward: {total_reward:.2f}")

        # --- FINAL SAVE ---
        agent.save_checkpoint(FINAL_MODEL_PATH, num_episodes)
        print(f"Training Complete. Final model saved to {FINAL_MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n>>> Ctrl+C detected. Saving checkpoint and exiting...")
        agent.save_checkpoint(CHECKPOINT_PATH, episode)
    finally:
        env.stop_simulation()
        print("Simulation stopped.")

if __name__ == "__main__":
    run_training()