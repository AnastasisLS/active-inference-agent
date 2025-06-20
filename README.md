# Active Inference Agent in a Simulated Environment

## Project Overview

This project implements an active inference agent in a simple control task (CartPole) to demonstrate how agents can minimize "surprise" (free energy) rather than maximizing rewards. The agent uses a learned probabilistic world model to make decisions, showcasing uncertainty-aware behavior compared to standard reinforcement learning agents.

### Key Features

- **Active Inference Agent**: Implements free energy minimization for decision making
- **DQN Baseline**: Traditional reinforcement learning agent for comparison
- **Uncertainty Quantification**: Belief state management and entropy analysis
- **Risk-Averse Behavior**: Demonstrates safer decision making under uncertainty
- **Comprehensive Analysis**: Performance, behavioral, and uncertainty metrics

### Core Technologies

- **Python 3.8+** with Jupyter notebooks
- **PyTorch** for neural network implementations
- **Gymnasium** for the CartPole environment
- **NumPy/SciPy** for numerical computations
- **Matplotlib/Seaborn** for visualization
- **PyMC** for probabilistic programming

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd active-inference-project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python test_setup.py
   ```

## Quick Start

### Running the Baseline DQN Agent

```python
from agents.dqn_agent import DQNAgent
from environments.cartpole_env import CartPoleEnv

# Initialize environment and agent
env = CartPoleEnv()
agent = DQNAgent(state_dim=4, action_dim=2)

# Train the agent
agent.train(env, episodes=1000)
```

### Running the Active Inference Agent

```python
from agents.active_inference_agent import ActiveInferenceAgent
from environments.cartpole_env import CartPoleEnv

# Initialize environment and agent
env = CartPoleEnv()
agent = ActiveInferenceAgent(state_dim=4, action_dim=2)

# Train the agent
agent.train(env, episodes=1000)
```

### Running Comparisons

```python
from utils.comparison import compare_agents

# Compare DQN vs Active Inference
results = compare_agents(['dqn', 'active_inference'], episodes=100)
```

## Project Structure

```
active_inference_project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── main.py                  # Main training script
├── config/                  # Configuration files
│   ├── agent_config.py      # Agent hyperparameters
│   ├── environment_config.py # Environment settings
│   └── training_config.py   # Training parameters
├── agents/                  # Agent implementations
│   ├── base_agent.py        # Base agent class
│   ├── dqn_agent.py         # DQN baseline agent
│   └── active_inference_agent.py # Active inference agent
├── environments/            # Environment wrappers
│   ├── cartpole_env.py      # CartPole environment
│   └── environment_wrapper.py # Base environment wrapper
├── models/                  # Model components
│   ├── generative_model.py  # Generative model
│   ├── transition_model.py  # State transition model
│   ├── observation_model.py # Observation model
│   └── belief_state.py      # Belief state management
├── utils/                   # Utility functions
│   ├── plotting.py          # Plotting utilities
│   ├── metrics.py           # Performance metrics
│   ├── visualization.py     # Visualization tools
│   └── data_logging.py      # Data logging utilities
├── notebooks/               # Jupyter notebooks
│   ├── 01_environment_exploration.ipynb
│   ├── 02_dqn_baseline.ipynb
│   ├── 03_active_inference_implementation.ipynb
│   ├── 04_comparison_analysis.ipynb
│   └── 05_results_visualization.ipynb
├── tests/                   # Test files
│   ├── test_agents.py       # Agent tests
│   ├── test_models.py       # Model tests
│   └── test_environments.py # Environment tests
└── data/                    # Data storage
    ├── logs/                # Training logs
    ├── models/              # Saved models
    └── results/             # Results and plots
```

## Usage Examples

### Training Agents

```python
# Train DQN agent
python main.py --agent dqn --episodes 1000 --save_model

# Train Active Inference agent
python main.py --agent active_inference --episodes 1000 --save_model

# Compare both agents
python main.py --compare --episodes 500
```

### Jupyter Notebooks

1. **Environment Exploration**: `notebooks/01_environment_exploration.ipynb`
   - Explore CartPole environment dynamics
   - Analyze state and action spaces

2. **DQN Baseline**: `notebooks/02_dqn_baseline.ipynb`
   - Implement and train DQN agent
   - Analyze performance and behavior

3. **Active Inference**: `notebooks/03_active_inference_implementation.ipynb`
   - Implement active inference components
   - Train and analyze active inference agent

4. **Comparison Analysis**: `notebooks/04_comparison_analysis.ipynb`
   - Compare DQN vs Active Inference
   - Analyze behavioral differences

5. **Results Visualization**: `notebooks/05_results_visualization.ipynb`
   - Create comprehensive visualizations
   - Generate final results and insights

## Results and Analysis

### Performance Comparison

The project compares the performance of:
- **DQN Agent**: Traditional reward-maximizing approach
- **Active Inference Agent**: Free energy minimization approach

### Key Metrics

1. **Performance Metrics:**
   - Episode success rate
   - Average episode length
   - Total reward accumulation
   - Learning speed

2. **Behavioral Metrics:**
   - Action distribution analysis
   - State visitation patterns
   - Policy consistency
   - Exploration patterns

3. **Uncertainty Metrics:**
   - Belief entropy over time
   - Confidence in predictions
   - Risk assessment accuracy

### Expected Findings

- Active inference agents demonstrate more conservative behavior under uncertainty
- Better handling of ambiguous situations
- More interpretable decision-making process
- Natural safety constraints through uncertainty awareness

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific test categories:

```bash
pytest tests/test_agents.py
pytest tests/test_models.py
pytest tests/test_environments.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Active Inference theory by Karl Friston
- OpenAI Gym/Gymnasium for the CartPole environment
- PyTorch community for deep learning tools
- PyMC community for probabilistic programming

## Contact

For questions or contributions, please open an issue on GitHub or contact the maintainers. 