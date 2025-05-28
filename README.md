# Adaptive DQN for Breakout

A Deep Q-Network (DQN) implementation with adaptive difficulty for the classic Breakout game. The environment dynamically adjusts its difficulty based on agent performance, providing a challenging and evolving training experience.

## Features

- **Adaptive Difficulty**: Environment automatically adjusts paddle speed, ball speed, and paddle size based on agent performance
- **Deep Q-Network**: Neural network-based reinforcement learning agent with experience replay and target networks
- **Visual Interface**: Interactive game visualization with real-time difficulty monitoring
- **Multiple Difficulty Levels**: Easy, Normal, Hard, and Extreme modes with different adaptation intensities
- **Performance Tracking**: Comprehensive metrics and visualization of training progress
- **Save/Load System**: Persistent storage of trained agents and training results

## Files

- `main.py` - Main training script with adaptive environment and DQN agent
- `interface.py` - Visual game interface for testing trained agents
- `README.md` - This documentation file

## Requirements

```bash
numpy
matplotlib
pickle (built-in)
json (built-in)
collections (built-in)
datetime (built-in)
time (built-in)
random (built-in)
```

## Quick Start

### Training a New Agent

```bash
python main.py
```



### Using the Visual Interface

```bash
python interface.py
```


Choose integration option:
1. Demo integration (rule-based agent)
2. Load from saved agent file

Select difficulty level (1-4) and enjoy the interactive game.

## Game Controls (Visual Interface)

- **Arrow Keys / A,D** - Move paddle (human mode)
- **T** - Toggle between AI and human control
- **1-4** - Change difficulty level during gameplay
- **R** - Restart game
- **S** - Show AI performance statistics
- **I** - Show detailed difficulty information
- **C** - Display difficulty menu
- **Q** - Quit game

## Adaptive Difficulty System

The environment monitors agent performance and adjusts difficulty in real-time:

### Performance Metrics
- Average score over last 100 episodes
- Paddle hits and brick destruction rate
- Episode length and completion rate

### Difficulty Adjustments
- **Paddle Speed**: Slower paddle makes the game harder
- **Ball Speed**: Faster ball increases challenge
- **Paddle Size**: Smaller paddle reduces control
- **Brick Regeneration**: Destroyed bricks can reappear

### Difficulty Levels

| Level | Description | Adaptation Intensity |
|-------|-------------|---------------------|
| Easy | Minimal changes, good for beginners | Low |
| Normal | Standard adaptive difficulty | Medium |
| Hard | Aggressive changes for experienced players | High |
| Extreme | Maximum chaos and challenge | Very High |

## Architecture

### DQN Agent
- **Neural Network**: 3-layer fully connected network with ReLU activation
- **Experience Replay**: Stores and samples past experiences for training
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy**: Exploration strategy with decaying epsilon

### Environment
- **State Space**: 84x84 grid representation of game state
- **Action Space**: 3 actions (no-op, move left, move right)
- **Reward System**: Points for paddle hits, brick destruction, and game completion
- **Adaptive Parameters**: Dynamic modification of game physics

## Training Process

1. **Initialization**: Create environment and agent with default parameters
2. **Episode Loop**: Agent interacts with environment, collecting experiences
3. **Experience Replay**: Train neural network on batches of stored experiences
4. **Target Network Update**: Periodically sync target network weights
5. **Difficulty Adaptation**: Adjust environment parameters based on performance
6. **Progress Tracking**: Monitor and visualize training metrics

## Performance Analysis

The system provides comprehensive performance analysis:

- **Episode Scores**: Track score progression over time
- **Moving Averages**: Smooth performance trends
- **Difficulty Progression**: Monitor adaptation over training
- **Training Loss**: Neural network learning progress

## File Outputs

### Saved Agent Files
- Format: `adaptive_dqn_agent.pkl`
- Contains: Trained neural network weights and agent parameters
- Usage: Load for continued training or deployment

### Results Files
- Format: `adaptive_dqn_results.json`
- Contains: Training metrics, performance data, and statistics
- Usage: Analysis and comparison of different training runs

## Customization

### Hyperparameters
Modify these values in the DQNAgent class:
- `learning_rate`: Neural network learning rate (default: 0.001)
- `epsilon_decay`: Exploration decay rate (default: 0.995)
- `gamma`: Discount factor for future rewards (default: 0.95)
- `batch_size`: Training batch size (default: 32)

### Environment Parameters
Adjust these in AdaptiveBreakoutEnvironment:
- `adaptation_frequency`: How often to adapt difficulty (default: 50 steps)
- `paddle_speed_range`: Min/max paddle speed modifiers
- `ball_speed_range`: Min/max ball speed modifiers
- `paddle_size_range`: Min/max paddle size modifiers

### Difficulty Levels
Customize difficulty configurations in the DifficultyLevels class:
- Modification frequencies
- Parameter ranges
- Adaptation intensities

## Troubleshooting

### Common Issues

**Training is too slow**
- Reduce number of episodes
- Decrease render frequency
- Use smaller neural network

**Agent performance is poor**
- Increase training episodes
- Adjust learning rate
- Modify reward structure

**Visual interface not responding**
- Check matplotlib backend
- Ensure proper key focus on game window
- Restart if controls become unresponsive

**File saving errors**
- Check write permissions in directory
- Ensure sufficient disk space
- Verify Python pickle/json modules

### Performance Tips

- Start with quick training (500 episodes) to test setup
- Use full training (2000+ episodes) for best results
- Monitor difficulty progression during training
- Save agents periodically during long training runs
- Test different difficulty levels to find optimal challenge

## Example Usage

\`\`\`python
# Train a new agent
from main import comprehensive_training

agent, tracker, env = comprehensive_training(episodes=1000)

# Load and test existing agent
from main import load_trained_agent, demo_trained_agent

agent = load_trained_agent("adaptive_dqn_agent_20240101_120000.pkl")
demo_trained_agent(agent, episodes=5)

# Use with visual interface
from interface import integrate_dqn_with_visual_game

integrate_dqn_with_visual_game(trained_agent=agent, difficulty_level='HARD')
\`\`\`

## Contributing

Feel free to modify and extend the code:
- Add new difficulty adaptation strategies
- Implement different neural network architectures
- Create additional performance metrics
- Develop new visualization features


