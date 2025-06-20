# GitHub Repository Setup Guide

## Quick Setup Instructions

### Option 1: Use the Setup Script (Recommended)
```bash
./setup_github.sh
```

### Option 2: Manual Setup

1. **Create GitHub Repository:**
   - Go to https://github.com/new
   - Repository name: `active-inference-agent` (or your preferred name)
   - Description: `Active Inference Agent in a Simulated Environment - Week 1 Implementation`
   - Make it Public or Private
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Add Remote and Push:**
   ```bash
   # Replace YOUR_USERNAME with your GitHub username
   git remote add origin https://github.com/YOUR_USERNAME/active-inference-agent.git
   git branch -M main
   git push -u origin main
   ```

## What's Included in This Repository

### Week 1 Implementation:
- ✅ **Project Structure**: Modular architecture with clear separation of concerns
- ✅ **Configuration System**: Flexible configuration for agents, environments, and training
- ✅ **Environment Wrapper**: CartPole environment with proper interfacing
- ✅ **DQN Baseline**: Deep Q-Network agent with experience replay
- ✅ **Training Infrastructure**: Complete training loop with logging
- ✅ **Utility Functions**: Metrics, plotting, visualization, and data logging
- ✅ **Documentation**: Comprehensive README and implementation plans
- ✅ **Testing**: Setup verification and basic functionality tests

### Key Files:
- `main.py` - Main training script with CLI interface
- `test_setup.py` - Setup verification script
- `notebooks/01_week1_demo.ipynb` - Jupyter notebook demonstration
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation

### Project Structure:
```
├── agents/           # Agent implementations (DQN baseline)
├── config/           # Configuration system
├── environments/     # Environment wrappers
├── utils/           # Utility functions (metrics, plotting, etc.)
├── notebooks/       # Jupyter notebooks
├── main.py          # Main training script
├── test_setup.py    # Setup verification
└── README.md        # Project documentation
```

## Next Steps

After pushing to GitHub:

1. **Test the Setup:**
   ```bash
   python test_setup.py
   ```

2. **Run the Demo:**
   ```bash
   python main.py --agent dqn --episodes 100
   ```

3. **Explore the Notebook:**
   ```bash
   jupyter notebook notebooks/01_week1_demo.ipynb
   ```

4. **Continue Development:**
   - Week 2-3: Implement Active Inference framework
   - Week 4-5: Advanced features and optimization
   - Week 6: Final comparison and documentation

## Repository Features

- 🚀 **Ready to Run**: All dependencies and setup included
- 📊 **Comprehensive Logging**: Training metrics and visualization
- 🔧 **Modular Design**: Easy to extend and modify
- 📚 **Well Documented**: Clear documentation and examples
- 🧪 **Tested**: Setup verification and basic functionality tests

Your Active Inference Agent project is now ready for development and collaboration! 