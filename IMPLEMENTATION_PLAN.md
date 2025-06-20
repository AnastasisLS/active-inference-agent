# Active Inference Agent in a Simulated Environment
## Detailed Implementation Plan

### Project Overview
This project implements an active inference agent in a simple control task (CartPole) to demonstrate how agents can minimize "surprise" (free energy) rather than maximizing rewards. The agent uses a learned probabilistic world model to make decisions, showcasing uncertainty-aware behavior compared to standard reinforcement learning agents.

### Core Technologies
- **Python 3.8+** with Jupyter notebooks
- **PyTorch** for neural network implementations
- **OpenAI Gym** for the CartPole environment
- **NumPy/SciPy** for numerical computations
- **Matplotlib/Seaborn** for visualization
- **PyMC** for probabilistic programming (alternative to pymdp)
- **Custom Active Inference Framework** (since pymdp may have limitations)

---

## Phase 1: Project Setup and Environment (Week 1)

### Step 1.1: Project Structure Setup
```
active_inference_project/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── main.py
├── config/
│   ├── __init__.py
│   ├── agent_config.py
│   ├── environment_config.py
│   └── training_config.py
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── dqn_agent.py
│   ├── active_inference_agent.py
│   └── comparison_agent.py
├── environments/
│   ├── __init__.py
│   ├── cartpole_env.py
│   └── environment_wrapper.py
├── models/
│   ├── __init__.py
│   ├── generative_model.py
│   ├── transition_model.py
│   ├── observation_model.py
│   └── belief_state.py
├── utils/
│   ├── __init__.py
│   ├── plotting.py
│   ├── metrics.py
│   ├── visualization.py
│   └── data_logging.py
├── notebooks/
│   ├── 01_environment_exploration.ipynb
│   ├── 02_dqn_baseline.ipynb
│   ├── 03_active_inference_implementation.ipynb
│   ├── 04_comparison_analysis.ipynb
│   └── 05_results_visualization.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_models.py
│   └── test_environments.py
└── data/
    ├── logs/
    ├── models/
    └── results/
```

### Step 1.2: Dependencies and Environment Setup
**requirements.txt:**
```
torch>=1.12.0
torchvision>=0.13.0
gymnasium>=0.26.0
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.4.0
jupyter>=1.0.0
pymc>=4.0.0
arviz>=0.12.0
tqdm>=4.64.0
wandb>=0.13.0
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
```

### Step 1.3: Configuration System
Create configuration classes for:
- Agent hyperparameters
- Environment settings
- Training parameters
- Model architectures

---

## Phase 2: Environment and Baseline Implementation (Week 1-2)

### Step 2.1: CartPole Environment Wrapper
**File: environments/cartpole_env.py**

**Implementation Details:**
1. **State Space Analysis:**
   - Cart position: [-4.8, 4.8]
   - Cart velocity: [-∞, ∞]
   - Pole angle: [-0.418, 0.418] radians
   - Pole angular velocity: [-∞, ∞]

2. **Action Space:**
   - Binary actions: {0, 1} (left, right)

3. **Environment Wrapper Features:**
   - State normalization
   - Reward shaping for baseline comparison
   - Episode termination conditions
   - State discretization for active inference

### Step 2.2: DQN Baseline Agent
**File: agents/dqn_agent.py**

**Implementation Details:**
1. **Network Architecture:**
   ```python
   class DQNNetwork(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=128):
           super().__init__()
           self.fc1 = nn.Linear(state_dim, hidden_dim)
           self.fc2 = nn.Linear(hidden_dim, hidden_dim)
           self.fc3 = nn.Linear(hidden_dim, action_dim)
           self.relu = nn.ReLU()
   ```

2. **Training Algorithm:**
   - Experience replay buffer
   - Target network updates
   - ε-greedy exploration
   - Loss: MSE between Q-values and target Q-values

3. **Hyperparameters:**
   - Learning rate: 0.001
   - Discount factor: 0.99
   - Replay buffer size: 10000
   - Batch size: 64
   - Target update frequency: 1000 steps

### Step 2.3: Performance Metrics and Logging
**File: utils/metrics.py**

**Metrics to Track:**
1. **Episode Performance:**
   - Episode length
   - Total reward
   - Success rate (episodes lasting > 195 steps)

2. **Training Metrics:**
   - Loss values
   - Q-value statistics
   - Exploration rate (ε)

3. **Behavioral Metrics:**
   - Action distribution
   - State visitation patterns
   - Policy entropy

---

## Phase 3: Active Inference Framework (Week 2-3)

### Step 3.1: Generative Model Implementation
**File: models/generative_model.py**

**Core Components:**

1. **State Representation:**
   ```python
   class StateSpace:
       def __init__(self, state_dim, discretization_levels):
           self.state_dim = state_dim
           self.discretization_levels = discretization_levels
           self.state_bins = self._create_state_bins()
   ```

2. **Transition Model:**
   ```python
   class TransitionModel(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=64):
           super().__init__()
           # Neural network to predict P(s'|s, a)
           self.transition_net = nn.Sequential(
               nn.Linear(state_dim + action_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, state_dim),
               nn.Softmax(dim=-1)
           )
   ```

3. **Observation Model:**
   ```python
   class ObservationModel(nn.Module):
       def __init__(self, state_dim, obs_dim, hidden_dim=64):
           super().__init__()
           # Neural network to predict P(o|s)
           self.observation_net = nn.Sequential(
               nn.Linear(state_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, obs_dim),
               nn.Softmax(dim=-1)
           )
   ```

### Step 3.2: Belief State Management
**File: models/belief_state.py**

**Implementation Details:**
1. **Belief State Representation:**
   - Probability distribution over states
   - Uncertainty quantification
   - Belief update mechanisms

2. **Bayesian Update:**
   ```python
   def update_belief(self, observation, action):
       # Predict: P(s_t) = Σ P(s_t|s_{t-1}, a_{t-1}) * P(s_{t-1})
       predicted_belief = self.predict_belief(action)
       
       # Update: P(s_t|o_t) ∝ P(o_t|s_t) * P(s_t)
       updated_belief = self.bayesian_update(observation, predicted_belief)
       
       return updated_belief
   ```

### Step 3.3: Free Energy Computation
**File: models/generative_model.py**

**Free Energy Components:**
1. **Accuracy Term:** -log P(o|s)
2. **Complexity Term:** KL divergence between predicted and preferred beliefs

```python
def compute_free_energy(self, observation, belief, preferred_belief):
    # Accuracy term
    accuracy = -torch.log(self.observation_model(observation, belief))
    
    # Complexity term
    complexity = torch.kl_div(belief, preferred_belief, reduction='batchmean')
    
    return accuracy + complexity
```

---

## Phase 4: Active Inference Agent Implementation (Week 3-4)

### Step 4.1: Active Inference Agent Core
**File: agents/active_inference_agent.py**

**Implementation Details:**

1. **Agent Architecture:**
   ```python
   class ActiveInferenceAgent:
       def __init__(self, state_dim, action_dim, config):
           self.generative_model = GenerativeModel(state_dim, action_dim)
           self.belief_state = BeliefState(state_dim)
           self.action_planner = ActionPlanner(action_dim)
           self.config = config
   ```

2. **Action Planning:**
   ```python
   def plan_action(self, current_belief, horizon=5):
       # Generate action sequences
       action_sequences = self.generate_action_sequences(horizon)
       
       # Evaluate each sequence using free energy
       free_energies = []
       for action_seq in action_sequences:
           future_beliefs = self.predict_future_beliefs(current_belief, action_seq)
           fe = self.compute_expected_free_energy(future_beliefs)
           free_energies.append(fe)
       
       # Select action with minimum expected free energy
       best_action_idx = torch.argmin(torch.stack(free_energies))
       return action_sequences[best_action_idx][0]
   ```

3. **Expected Free Energy:**
   ```python
   def compute_expected_free_energy(self, future_beliefs):
       # EFE = E[G] = E[log P(o)] - E[log P(o|s)]
       # Where G is the expected free energy
       
       # Epistemic value: information gain
       epistemic_value = self.compute_epistemic_value(future_beliefs)
       
       # Pragmatic value: expected utility
       pragmatic_value = self.compute_pragmatic_value(future_beliefs)
       
       return epistemic_value + pragmatic_value
   ```

### Step 4.2: Uncertainty-Aware Decision Making
**File: agents/active_inference_agent.py**

**Uncertainty Handling:**
1. **Belief Entropy:**
   ```python
   def compute_belief_entropy(self, belief):
       return -torch.sum(belief * torch.log(belief + 1e-8))
   ```

2. **Risk-Averse Behavior:**
   - Higher uncertainty leads to more conservative actions
   - Exploration vs. exploitation balance
   - Safety constraints in action selection

### Step 4.3: Model Learning and Adaptation
**File: models/generative_model.py**

**Learning Mechanisms:**
1. **Transition Model Learning:**
   ```python
   def update_transition_model(self, state, action, next_state):
       # Update transition probabilities using experience
       target = self.encode_state(next_state)
       prediction = self.transition_model(state, action)
       loss = F.cross_entropy(prediction, target)
       
       self.transition_optimizer.zero_grad()
       loss.backward()
       self.transition_optimizer.step()
   ```

2. **Observation Model Learning:**
   ```python
   def update_observation_model(self, state, observation):
       # Update observation probabilities
       target = self.encode_observation(observation)
       prediction = self.observation_model(state)
       loss = F.cross_entropy(prediction, target)
       
       self.observation_optimizer.zero_grad()
       loss.backward()
       self.observation_optimizer.step()
   ```

---

## Phase 5: Training and Optimization (Week 4-5)

### Step 5.1: Training Pipeline
**File: main.py**

**Training Process:**
1. **Environment Interaction:**
   ```python
   def train_episode(agent, environment, max_steps=500):
       state = environment.reset()
       total_reward = 0
       episode_data = []
       
       for step in range(max_steps):
           # Agent action selection
           action = agent.select_action(state)
           
           # Environment step
           next_state, reward, done, info = environment.step(action)
           
           # Store experience
           episode_data.append({
               'state': state,
               'action': action,
               'reward': reward,
               'next_state': next_state,
               'done': done
           })
           
           # Update agent
           agent.update(state, action, reward, next_state, done)
           
           state = next_state
           total_reward += reward
           
           if done:
               break
       
       return total_reward, episode_data
   ```

2. **Hyperparameter Tuning:**
   - Learning rates for different model components
   - Planning horizon length
   - Belief update frequency
   - Exploration parameters

### Step 5.2: Performance Optimization
**File: utils/optimization.py**

**Optimization Techniques:**
1. **Parallel Planning:**
   - Use multiple planning threads
   - GPU acceleration for belief updates

2. **Approximate Inference:**
   - Variational methods for belief updates
   - Particle filtering for complex environments

3. **Memory Management:**
   - Efficient belief state storage
   - Experience replay for model learning

---

## Phase 6: Comparison and Analysis (Week 5-6)

### Step 6.1: Comprehensive Evaluation
**File: utils/comparison.py**

**Evaluation Metrics:**

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

### Step 6.2: Uncertainty Analysis
**File: utils/uncertainty_analysis.py**

**Uncertainty Quantification:**
1. **Belief State Analysis:**
   ```python
   def analyze_uncertainty_trajectory(agent, episode_data):
       uncertainties = []
       for step_data in episode_data:
           belief = agent.get_belief_state(step_data['state'])
           entropy = compute_entropy(belief)
           uncertainties.append(entropy)
       return uncertainties
   ```

2. **Risk-Averse Behavior Detection:**
   - Compare action choices under high vs. low uncertainty
   - Analyze conservative behavior patterns
   - Measure safety margin in decision making

### Step 6.3: Visualization and Interpretation
**File: utils/visualization.py**

**Visualization Components:**

1. **Trajectory Comparison:**
   ```python
   def plot_trajectory_comparison(dqn_trajectories, ai_trajectories):
       # Plot state space trajectories
       # Highlight different behavioral patterns
       # Show uncertainty regions
   ```

2. **Belief State Visualization:**
   ```python
   def plot_belief_evolution(belief_history):
       # Heatmap of belief states over time
       # Uncertainty visualization
       # Preferred state highlighting
   ```

3. **Decision Analysis:**
   ```python
   def plot_decision_analysis(agent_decisions):
       # Action selection patterns
       # Free energy landscape
       # Uncertainty-action correlation
   ```

---

## Phase 7: Documentation and Results (Week 6)

### Step 7.1: Comprehensive Documentation
**File: README.md**

**Documentation Sections:**
1. **Project Overview:**
   - Active inference principles
   - Implementation approach
   - Key contributions

2. **Installation and Setup:**
   - Environment setup
   - Dependencies installation
   - Quick start guide

3. **Usage Examples:**
   - Training agents
   - Running comparisons
   - Analyzing results

4. **Results and Analysis:**
   - Performance comparisons
   - Behavioral insights
   - Uncertainty analysis

### Step 7.2: Results Summary
**File: results_summary.md**

**Key Findings:**
1. **Performance Comparison:**
   - DQN vs. Active Inference performance
   - Learning curves and convergence
   - Success rates and stability

2. **Behavioral Differences:**
   - Exploration patterns
   - Risk assessment
   - Decision-making under uncertainty

3. **Uncertainty Handling:**
   - Belief state evolution
   - Confidence calibration
   - Safety implications

### Step 7.3: Code Quality and Testing
**File: tests/**

**Testing Strategy:**
1. **Unit Tests:**
   - Agent functionality
   - Model components
   - Utility functions

2. **Integration Tests:**
   - End-to-end training
   - Environment interactions
   - Comparison workflows

3. **Performance Tests:**
   - Training speed
   - Memory usage
   - Scalability

---

## Implementation Timeline

### Week 1:
- [ ] Project structure setup
- [ ] Environment implementation
- [ ] DQN baseline agent
- [ ] Basic training pipeline

### Week 2:
- [ ] Generative model framework
- [ ] Belief state management
- [ ] Free energy computation
- [ ] Initial active inference agent

### Week 3:
- [ ] Action planning implementation
- [ ] Uncertainty handling
- [ ] Model learning mechanisms
- [ ] Training optimization

### Week 4:
- [ ] Comprehensive training
- [ ] Hyperparameter tuning
- [ ] Performance optimization
- [ ] Initial results collection

### Week 5:
- [ ] Comparison analysis
- [ ] Uncertainty quantification
- [ ] Behavioral analysis
- [ ] Visualization development

### Week 6:
- [ ] Final results analysis
- [ ] Documentation completion
- [ ] Code cleanup and testing
- [ ] Repository preparation

---

## Success Criteria

### Technical Success:
1. **Functional Implementation:**
   - Active inference agent successfully learns and operates
   - Generative model accurately represents environment
   - Belief updates work correctly

2. **Performance Metrics:**
   - Agent achieves comparable performance to DQN
   - Demonstrates uncertainty-aware behavior
   - Shows risk-averse decision making

3. **Code Quality:**
   - Well-documented and tested code
   - Modular and extensible architecture
   - Reproducible results

### Research Success:
1. **Novel Insights:**
   - Clear behavioral differences between agents
   - Uncertainty handling demonstration
   - Safety implications analysis

2. **Educational Value:**
   - Clear explanation of active inference
   - Practical implementation guide
   - Comparative analysis framework

3. **Portfolio Impact:**
   - Demonstrates advanced ML concepts
   - Shows probabilistic thinking
   - Highlights interpretability and safety

---

## Risk Mitigation

### Technical Risks:
1. **Implementation Complexity:**
   - Start with simplified models
   - Incremental development
   - Extensive testing

2. **Performance Issues:**
   - Optimize critical components
   - Use efficient data structures
   - Profile and benchmark

3. **Convergence Problems:**
   - Careful hyperparameter tuning
   - Multiple training runs
   - Robust evaluation metrics

### Timeline Risks:
1. **Scope Creep:**
   - Focus on core functionality
   - Prioritize essential features
   - Regular progress reviews

2. **Technical Blockers:**
   - Alternative implementation approaches
   - Community resources and forums
   - Iterative problem solving

---

## Future Extensions

### Short-term Enhancements:
1. **Additional Environments:**
   - MountainCar
   - Acrobot
   - LunarLander

2. **Advanced Models:**
   - Hierarchical active inference
   - Continuous state spaces
   - Multi-agent scenarios

### Long-term Research:
1. **Theoretical Contributions:**
   - Novel free energy formulations
   - Improved planning algorithms
   - Uncertainty quantification methods

2. **Applications:**
   - Robotics control
   - Autonomous systems
   - Human-AI interaction

This detailed plan provides a comprehensive roadmap for implementing an active inference agent that demonstrates uncertainty-aware decision making compared to traditional reinforcement learning approaches. 