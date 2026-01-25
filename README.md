# Multi-Agent Deep Reinforcement Learning (MADRL) for HVAC Control

A multi-agent deep Q-learning (DDQN) system for optimizing heating, ventilation, and air conditioning (HVAC) control in a 5-zone building using EnergyPlus simulations.

## Overview

This project implements a multi-agent reinforcement learning framework to learn optimal setpoint control for HVAC systems in buildings. Each zone is controlled by an independent agent that learns through interaction with an EnergyPlus building simulation environment.

### Key Features
- **Multi-Agent Architecture**: Independent DDQN agents for 5 building zones (SPACE1-1 to SPACE5-1)
- **3-Phase Training Approach**: 
  - Phase 1: Pretraining with broadcasting (40 episodes)
  - Phase 2: Parameter sharing (50 episodes)
  - Phase 3: Independent learning (100 episodes)
- **Building Simulation**: Integration with EnergyPlus for realistic thermal modeling
- **Dual Optimization**: Balances energy consumption and occupant comfort (thermal comfort complaints)
- **Real Weather Data**: Uses a 15-year weather database (Rabat-Sale, Morocco)

## Project Structure

```
madrl/
├── main.py                          # Main training loop and orchestration
├── agent.py                         # DDQN agent implementation
├── hvac_env.py                      # EnergyPlus environment wrapper
├── occupancy.py                     # Occupancy management and comfort logic
├── setup_simulation.py              # IDF file configuration and validation
├── 5ZoneAirCooled.idf              # EnergyPlus building model
├── MAR_RK_Rabat-Sale.AP.*.epw      # Weather file (2009-2023)
├── plot_results.py                  # Visualization utilities
├── plots.py                         # Training metrics plotting
├── logs/                            # Training metrics (CSV format)
│   ├── step_metrics.csv            # Per-step data (temp, energy, complaints)
│   └── episode_metrics.csv         # Per-episode aggregated metrics
├── checkpoints_logging/             # Saved model weights
│   ├── agent_SPACE*.pth            # Model checkpoints for each zone
│   └── training_state.json         # Training progress tracker
└── results_logging/                 # Detailed episode outputs
    └── ep_out_{episode}_{timestamp}/ # Per-episode results
```

## Requirements

### Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Gymnasium
- EnergyPlus 24.1.0
- PyEnergyPlus API

### Installation
pip install -r requirements.txt
2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch numpy gymnasium
   ```

4. **Install EnergyPlus**
   - Download and install EnergyPlus 24.1.0 from [EnergyPlus website](https://energyplus.net/)
   - Update the `energyplus_install_path` in `hvac_env.py` to point to your installation

5. **Verify setup**
   ```bash
   python setup_simulation.py
   ```

## Training

### Quick Start

Run the complete training pipeline:
```bash
python main.py
```

The training manager will:
- Resume from the last checkpoint if interrupted
- Execute all 3 phases sequentially
- Log metrics at each step and episode
- Save model checkpoints after each episode

### Training Configuration

Key hyperparameters (in `agent.py`):
- `GAMMA = 0.9` - Discount factor
- `BATCH_SIZE = 256` - Batch size for neural network updates
- `BUFFER_SIZE = 576` - Experience replay buffer capacity
- `TARGET_UPDATE_FREQ = 2` - Target network update frequency (episodes)

Reward function parameters (in `hvac_env.py`):
- `lambda_e = 0.008` - Energy consumption weight
- `lambda_m = 0.12` - Comfort complaints weight
- `l_clip = 5.0` - Clipping threshold for outliers

### Monitoring Training

View real-time metrics:
```bash
python plots.py
```

This generates plots showing:
- Episode rewards over time
- Energy consumption trends
- Occupant complaint trends
- Per-zone performance

## Agent Architecture

### DDQN Network
```
Input (10 features) 
  ↓
Linear(256) → LayerNorm → ReLU
  ↓
Linear(256) → LayerNorm → ReLU
  ↓
Linear(11 actions)
```

### State Space (10 features)
- Day of year
- Minute of day
- Week of year
- Outdoor temperature
- Solar radiation (horizontal)
- Solar radiation (south)
- Zone temperature
- Zone occupancy
- Additional contextual features

### Action Space
- 11 discrete actions
- Mapping: Setpoint temperatures from 15°C to 25°C (1°C increments)

## EnergyPlus Building Model

The `5ZoneAirCooled.idf` file represents:
- 5 thermally conditioned zones
- CAV (Constant Air Volume) HVAC system
- Variable setpoint control (15°C - 25°C)
- Realistic occupancy profiles
- Building envelope thermal properties

## Utilities

### Visualization
```bash
python plot_results.py  # Generate training analysis plots
```

### Simulation Tools
- `fix_idf.py` - Corrects IDF file syntax issues
- `fix_duration.py` - Adjusts simulation duration
- `force_date.py` - Manages simulation date handling

## Logging and Outputs

### Step Metrics (`logs/step_metrics.csv`)
Logged every step with columns:
- Phase, Episode, Step, Zone
- OutdoorTemp, ZoneTemp, Occupancy
- HeatingSetpoint, Energy_J, Complaint, Reward

### Episode Metrics (`logs/episode_metrics.csv`)
Aggregated per-episode with columns:
- Phase, Episode
- TotalReward, TotalEnergy_kWh, TotalComplaints

### Episode Results (`results_logging/`)
Each episode generates detailed output files containing:
- EnergyPlus simulation logs
- Zone-by-zone metrics
- Control actions taken

## Model Checkpoints

Trained model weights are saved in `checkpoints_logging/`:
- `agent_SPACE{1-5}-1.pth` - Per-zone model weights
- `training_state.json` - Current phase and episode number

### Loading Checkpoints
```python
agent = Agent(input_dim=10, action_dim=11)
agent.policy_net.load_state_dict(torch.load('checkpoints_logging/agent_SPACE1-1.pth'))
```

## Results and Analysis

### Key Metrics
1. **Cumulative Reward** - Combined signal of energy and comfort
2. **Energy Consumption** - Measured in kWh per episode
3. **Thermal Comfort Complaints** - Penalizes temperature violations

### Expected Performance
- Phase 1: Rapid improvement as agents learn basic control
- Phase 2: Continued improvement with parameter sharing across zones
- Phase 3: Fine-tuning and independent agent specialization

## Troubleshooting

### EnergyPlus Connection Issues
- Verify EnergyPlus installation path in `hvac_env.py`
- Check that `5ZoneAirCooled.idf` and weather file exist
- Run `setup_simulation.py` to validate IDF file

### Training Crashes
- Check available GPU memory (if using CUDA)
- Review EnergyPlus error logs in `results_logging/`
- Verify weather file compatibility with simulation dates

### Memory Issues
- Reduce `BUFFER_SIZE` in `agent.py`
- Lower `BATCH_SIZE` for smaller GPU/RAM systems

## Performance Notes

- **Episode Duration**: ~2-5 minutes per episode depending on hardware
- **Total Training Time**: ~5-15 hours for all 190 episodes
- **Recommended**: GPU acceleration (CUDA) for faster training

## Future Enhancements

- Policy gradient methods (A2C, PPO)
- Attention mechanisms for inter-zone learning
- Multi-objective optimization (Pareto frontier)
- Real building deployment and validation
- Demand response integration

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Changes are tested with `main.py`
- Training metrics are logged properly

## License

[Specify your license here]

## Contact

For questions or issues, please contact the project maintainers.

---

**Last Updated**: January 2026  
**Python Version**: 3.8+  
**EnergyPlus Version**: 24.1.0
