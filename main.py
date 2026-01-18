import os
import torch
import numpy as np
from hvac_env import EnergyPlusEnv
from agent import Agent
from occupancy import OccupancyManager

ZONE_NAMES = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
OUTPUT_DIR = "results_paper_impl"

# Paper Config
EPISODES_PRETRAIN = 10  # Reduced for testing (Paper says 12,500!)
EPISODES_SHARED = 10    # Reduced for testing
EPISODES_INDIE = 5      # Reduced for testing

def train_framework():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    env = EnergyPlusEnv("5ZoneAirCooled.idf", "MAR_RK_Rabat-Sale.AP.601350_TMYx.2009-2023.epw", OUTPUT_DIR)
    env.occupancy_manager = OccupancyManager(ZONE_NAMES)
    
    # --- PHASE 1: Pretraining (Broadcasting Mode) ---
    print("\n>>> STARTING PHASE 1: PRETRAINING (Broadcasting) <<<")
    # Only ONE agent exists.
    master_agent = Agent(input_dim=10, action_dim=11)
    
    for ep in range(EPISODES_PRETRAIN):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # We pick ONE action based on Zone 1's state
            # In true broadcasting, we might average states, but paper implies using one agent's perspective
            state = obs[ZONE_NAMES[0]]
            action_idx = master_agent.get_action(state)
            
            # Broadcast action to ALL zones
            actions = {z: action_idx for z in ZONE_NAMES}
            
            next_obs, reward, done, _, _ = env.step(actions)
            
            # Store experience (State Z1, Action, Global Reward, Next State Z1)
            next_state = next_obs[ZONE_NAMES[0]]
            master_agent.store_transition(state, action_idx, reward, next_state, done)
            master_agent.update()
            
            obs = next_obs
            total_reward += reward
            
        print(f"Phase 1 - Episode {ep+1}: Reward {total_reward:.2f}")

    # --- PHASE 2: Semi-Multi-Agent (Parameter Sharing) ---
    print("\n>>> STARTING PHASE 2: PARAMETER SHARING <<<")
    # Initialize 5 agents, copying weights from master
    agents = {}
    for zone in ZONE_NAMES:
        agents[zone] = Agent(input_dim=10, action_dim=11)
        agents[zone].copy_weights_from(master_agent)
        
    for ep in range(EPISODES_SHARED):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            actions = {}
            # 1. Inference
            for zone in ZONE_NAMES:
                actions[zone] = agents[zone].get_action(obs[zone])
                
            # 2. Step
            next_obs, reward, done, _, _ = env.step(actions)
            
            # 3. Store & Update
            # In Parameter Sharing, all agents push to their buffers, 
            # BUT we force them to stay identical.
            # A common implementation of "Parameter Sharing" is physically using ONE network object
            # passed to 5 inputs. Let's do that to strictly adhere to the concept.
            
            # We use agents['SPACE1-1'] as the shared brain
            shared_brain = agents[ZONE_NAMES[0]]
            
            for zone in ZONE_NAMES:
                shared_brain.store_transition(obs[zone], actions[zone], reward, next_obs[zone], done)
            
            shared_brain.update() # Updates weights based on mixed batch
            
            obs = next_obs
            total_reward += reward
            
        print(f"Phase 2 - Episode {ep+1}: Reward {total_reward:.2f}")

    # --- PHASE 3: Independent Learning ---
    print("\n>>> STARTING PHASE 3: INDEPENDENT LEARNING <<<")
    # Now we decouple. We distribute the trained shared_brain to everyone.
    for zone in ZONE_NAMES:
        if zone != ZONE_NAMES[0]:
            agents[zone].copy_weights_from(agents[ZONE_NAMES[0]])
            
    for ep in range(EPISODES_INDIE):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            actions = {}
            for zone in ZONE_NAMES:
                actions[zone] = agents[zone].get_action(obs[zone])
                
            next_obs, reward, done, _, _ = env.step(actions)
            
            for zone in ZONE_NAMES:
                agents[zone].store_transition(obs[zone], actions[zone], reward, next_obs[zone], done)
                agents[zone].update() # Update INDIVIDUALLY now
            
            obs = next_obs
            total_reward += reward
            
        print(f"Phase 3 - Episode {ep+1}: Reward {total_reward:.2f}")

    env.stop_simulation()
    
    # Save Final Models
    for zone in ZONE_NAMES:
        agents[zone].save(f"{OUTPUT_DIR}/model_{zone}.pth")

if __name__ == "__main__":
    train_framework()