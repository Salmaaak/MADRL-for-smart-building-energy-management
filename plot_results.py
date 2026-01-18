import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
LOG_DIR = "logs"
EPISODE_FILE = os.path.join(LOG_DIR, "episode_metrics.csv")
STEP_FILE = os.path.join(LOG_DIR, "step_metrics.csv")

def plot_learning_curves():
    if not os.path.exists(EPISODE_FILE):
        print(f"File not found: {EPISODE_FILE}. Run main.py first.")
        return

    df = pd.read_csv(EPISODE_FILE)
    
    # Set style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Total Reward
    sns.lineplot(ax=axes[0], data=df, x="Episode", y="Total_Reward", marker="o", color="b")
    axes[0].set_title("Learning Curve: Total Reward per Episode")
    axes[0].set_ylabel("Total Reward (Higher is Better)")

    # 2. Total Energy
    # Convert Joules to kWh for readability (1 J = 2.77e-7 kWh)
    df["Energy_kWh"] = df["Total_Energy_J"] / 3600000.0
    sns.lineplot(ax=axes[1], data=df, x="Episode", y="Energy_kWh", marker="o", color="orange")
    axes[1].set_title("Energy Consumption per Episode")
    axes[1].set_ylabel("Energy (kWh)")

    # 3. Total Complaints
    sns.lineplot(ax=axes[2], data=df, x="Episode", y="Total_Complaints", marker="o", color="r")
    axes[2].set_title("Comfort Violations per Episode")
    axes[2].set_ylabel("Total Complaint Score")
    axes[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()

def plot_last_episode_details():
    if not os.path.exists(STEP_FILE):
        print(f"File not found: {STEP_FILE}")
        return

    df = pd.read_csv(STEP_FILE)
    
    # Check if data exists
    if df.empty:
        print("Step metrics file is empty.")
        return

    # Filter for the LAST recorded episode
    last_episode = df["Episode"].max()
    episode_data = df[df["Episode"] == last_episode].copy()
    
    # Create a proper Time index for plotting (Step count usually suffices, but nice to see)
    episode_data = episode_data.reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- Primary Y-Axis: Temperature ---
    color = 'tab:blue'
    ax1.set_xlabel('Simulation Step (Time)')
    ax1.set_ylabel('Mean Zone Temperature (°C)', color=color)
    ax1.plot(episode_data.index, episode_data["Mean_Zone_Temp"], color=color, label="Zone Temp")
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add Comfort Band Reference (21°C +/- 1)
    ax1.axhline(y=21.0, color='green', linestyle='--', alpha=0.5, label="Target (21°C)")
    ax1.axhline(y=20.0, color='green', linestyle=':', alpha=0.3)
    ax1.axhline(y=22.0, color='green', linestyle=':', alpha=0.3)

    # --- Secondary Y-Axis: Occupancy ---
    ax2 = ax1.twinx()  
    color = 'tab:gray'
    ax2.set_ylabel('Max Occupancy (People)', color=color)  
    ax2.fill_between(episode_data.index, episode_data["Max_Occupancy"], color=color, alpha=0.2, label="Occupancy")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(episode_data["Max_Occupancy"].max(), 1) * 1.5) # Scale slightly higher

    # Title
    plt.title(f"Detailed View: Episode {last_episode} (Temp vs Occupancy)")
    fig.tight_layout()  
    plt.show()

    # --- Separate Plot for Energy Spikes ---
    plt.figure(figsize=(12, 4))
    plt.plot(episode_data.index, episode_data["Energy_J"], color='orange', linewidth=1)
    plt.title(f"Energy Consumption Profile: Episode {last_episode}")
    plt.ylabel("Energy (Joules)")
    plt.xlabel("Simulation Step")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    print("Generating Learning Curves...")
    plot_learning_curves()
    
    print("Generating Detailed View of Last Episode...")
    plot_last_episode_details()