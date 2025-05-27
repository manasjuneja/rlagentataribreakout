import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random
import time
from collections import deque
import pickle

# First, let's create a compatible DQN agent class that works with the visual game
class VisualGameCompatibleDQN:
    """DQN Agent adapted to work with the visual game interface"""
    
    def __init__(self, trained_agent=None):
        if trained_agent is not None:
            # Use a pre-trained agent
            self.q_network = trained_agent.q_network
            self.epsilon = 0.01  # Low epsilon for demonstration (mostly exploit)
            self.action_size = trained_agent.action_size
            self.is_trained = True
            print("✅ Loaded pre-trained DQN agent")
        else:
            # Create a simple agent for demonstration
            self.q_network = None
            self.epsilon = 0.1
            self.action_size = 3
            self.is_trained = False
            print("⚠️ Using simple rule-based agent (no DQN loaded)")
    
    def convert_visual_state_to_dqn_format(self, visual_game_state):
        """Convert the visual game state to DQN-compatible format"""
        
        # Visual game state format: [paddle_x, ball_x, ball_y, ball_dx, ball_dy, brick_density]
        # DQN expects: flattened 84x84 grid or similar
        
        if not self.is_trained:
            # For demo purposes, return the simple state
            return visual_game_state
        
        # Create a simplified grid representation for the DQN
        grid_size = 84
        state_grid = np.zeros((grid_size, grid_size))
        
        # Normalize coordinates to grid size
        paddle_x = int(visual_game_state[0] * grid_size)
        ball_x = int(visual_game_state[1] * grid_size)
        ball_y = int(visual_game_state[2] * grid_size)
        
        # Ensure coordinates are within bounds
        paddle_x = max(0, min(grid_size-1, paddle_x))
        ball_x = max(0, min(grid_size-1, ball_x))
        ball_y = max(0, min(grid_size-1, ball_y))
        
        # Place paddle in grid (bottom area)
        paddle_y = grid_size - 5
        for i in range(max(0, paddle_x-5), min(grid_size, paddle_x+5)):
            for j in range(max(0, paddle_y-2), min(grid_size, paddle_y+2)):
                state_grid[j, i] = 1.0  # Paddle value
        
        # Place ball in grid
        for i in range(max(0, ball_x-1), min(grid_size, ball_x+1)):
            for j in range(max(0, ball_y-1), min(grid_size, ball_y+1)):
                state_grid[j, i] = 2.0  # Ball value
        
        # Add brick information (simplified)
        brick_density = visual_game_state[5]
        # Fill top area with brick representation
        for i in range(0, grid_size, 8):
            for j in range(0, 20):
                if random.random() < brick_density:
                    state_grid[j, i] = 3.0  # Brick value
        
        return state_grid.flatten()
    
    def act(self, visual_game_state, record_strategy=False):
        """Get action from DQN agent given visual game state"""
        
        if not self.is_trained or self.q_network is None:
            # Fallback to simple rule-based behavior
            return self.simple_rule_based_action(visual_game_state)
        
        # Convert state format
        dqn_state = self.convert_visual_state_to_dqn_format(visual_game_state)
        
        # Use epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Get Q-values from network
        state_reshaped = dqn_state.reshape(1, -1)
        q_values = self.q_network.forward(state_reshaped)
        action = np.argmax(q_values[0])
        
        return action
    
    def simple_rule_based_action(self, state):
        """Simple rule-based action for when DQN is not available"""
        paddle_x = state[0]
        ball_x = state[1]
        ball_y = state[2]
        ball_dx = state[3]
        ball_dy = state[4]
        
        # Predict where ball will be
        if ball_dy > 0:  # Ball moving down
            predicted_x = ball_x + (ball_dx * 0.2)
        else:
            predicted_x = ball_x
        
        # Move towards predicted position
        if predicted_x < paddle_x - 0.05:
            return 1  # Move left
        elif predicted_x > paddle_x + 0.05:
            return 2  # Move right
        else:
            return 0  # Stay

# Enhanced Visual Game with DQN Integration
class DQNVisualBreakout:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.reset_game()
        
        # Game objects
        self.paddle_width = 80
        self.paddle_height = 15
        self.ball_radius = 8
        self.brick_width = 75
        self.brick_height = 20
        
        # Colors
        self.colors = {
            'background': '#000000',
            'paddle': '#FFFFFF',
            'ball': '#FF6B6B',
            'bricks': ['#FF9F43', '#10AC84', '#5F27CD', '#00D2D3', '#FF3838', '#2E86AB'],
            'text': '#FFFFFF'
        }
        
        # Game state tracking
        self.game_stats = {
            'score': 0,
            'lives': 3,
            'level': 1,
            'paddle_hits': 0,
            'bricks_destroyed': 0,
            'game_time': 0
        }
        
        # DQN agent
        self.dqn_agent = None
        self.ai_mode = False
        
    def reset_game(self):
        """Reset the game to initial state"""
        # Paddle position
        self.paddle_x = self.width // 2 - 40
        self.paddle_y = self.height - 50
        
        # Ball position and velocity
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = random.choice([-4, 4])
        self.ball_dy = -4
        
        # Create bricks
        self.bricks = []
        brick_start_y = 80
        for row in range(6):
            for col in range(10):
                brick_x = col * (self.brick_width + 5) + 50
                brick_y = row * (self.brick_height + 5) + brick_start_y
                color_idx = row % len(self.colors['bricks'])
                self.bricks.append({
                    'x': brick_x,
                    'y': brick_y,
                    'color': self.colors['bricks'][color_idx],
                    'active': True,
                    'points': (6 - row) * 10
                })
        
        # Reset stats
        self.game_stats.update({
            'score': 0,
            'lives': 3,
            'level': 1,
            'paddle_hits': 0,
            'bricks_destroyed': 0,
            'game_time': 0
        })
        
        self.game_over = False
        self.game_won = False
    
    def get_state_for_dqn(self):
        """Get game state in format suitable for DQN"""
        # Normalize all values to 0-1 range
        state = np.array([
            self.paddle_x / self.width,                    # Paddle position
            self.ball_x / self.width,                      # Ball X position  
            self.ball_y / self.height,                     # Ball Y position
            self.ball_dx / 10.0,                          # Ball X velocity (normalized)
            self.ball_dy / 10.0,                          # Ball Y velocity (normalized)
            len([b for b in self.bricks if b['active']]) / len(self.bricks)  # Brick density
        ])
        return state
    
    def set_dqn_agent(self, dqn_agent):
        """Set the DQN agent"""
        self.dqn_agent = dqn_agent
        print(f"✅ DQN Agent set: {'Trained' if dqn_agent.is_trained else 'Rule-based'}")
    
    def update_game(self, action=None):
        """Update game state"""
        if self.game_over or self.game_won:
            return
        
        # Get action from DQN if in AI mode
        if self.ai_mode and self.dqn_agent is not None and action is None:
            state = self.get_state_for_dqn()
            action = self.dqn_agent.act(state)
        
        # Apply action
        if action == 1 and self.paddle_x > 0:  # Move left
            self.paddle_x -= 8
        elif action == 2 and self.paddle_x < self.width - self.paddle_width:  # Move right
            self.paddle_x += 8
        
        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= self.ball_radius or self.ball_x >= self.width - self.ball_radius:
            self.ball_dx = -self.ball_dx
        
        if self.ball_y <= self.ball_radius:
            self.ball_dy = -self.ball_dy
        
        # Ball collision with paddle
        if (self.ball_y + self.ball_radius >= self.paddle_y and
            self.ball_y - self.ball_radius <= self.paddle_y + self.paddle_height and
            self.ball_x >= self.paddle_x and
            self.ball_x <= self.paddle_x + self.paddle_width):
            
            # Calculate bounce angle
            hit_pos = (self.ball_x - self.paddle_x) / self.paddle_width
            angle_factor = (hit_pos - 0.5) * 2
            
            self.ball_dy = -abs(self.ball_dy)
            self.ball_dx += angle_factor * 2
            
            # Limit speed
            speed = np.sqrt(self.ball_dx**2 + self.ball_dy**2)
            if speed > 8:
                self.ball_dx = (self.ball_dx / speed) * 8
                self.ball_dy = (self.ball_dy / speed) * 8
            
            self.game_stats['paddle_hits'] += 1
            self.game_stats['score'] += 1
        
        # Ball collision with bricks
        for brick in self.bricks:
            if not brick['active']:
                continue
                
            if (self.ball_x + self.ball_radius >= brick['x'] and
                self.ball_x - self.ball_radius <= brick['x'] + self.brick_width and
                self.ball_y + self.ball_radius >= brick['y'] and
                self.ball_y - self.ball_radius <= brick['y'] + self.brick_height):
                
                brick['active'] = False
                self.ball_dy = -self.ball_dy
                self.game_stats['score'] += brick['points']
                self.game_stats['bricks_destroyed'] += 1
                break
        
        # Check if ball is lost
        if self.ball_y > self.height:
            self.game_stats['lives'] -= 1
            if self.game_stats['lives'] <= 0:
                self.game_over = True
            else:
                # Reset ball
                self.ball_x = self.width // 2
                self.ball_y = self.height // 2
                self.ball_dx = random.choice([-4, 4])
                self.ball_dy = -4
        
        # Check win condition
        if all(not brick['active'] for brick in self.bricks):
            self.game_won = True
        
        self.game_stats['game_time'] += 1

# Complete Integration Interface
class DQNGameInterface:
    def __init__(self):
        self.game = DQNVisualBreakout()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(0, self.game.width)
        self.ax.set_ylim(0, self.game.height)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('black')
        
        # Drawing objects
        self.paddle_rect = None
        self.ball_circle = None
        self.brick_rects = []
        self.text_objects = []
        
        # Animation control
        self.animation = None
        self.is_running = False
        self.frame_count = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.action_history = deque(maxlen=1000)
        
    def load_trained_dqn_agent(self, agent_file_path=None, agent_object=None):
        """Load a trained DQN agent"""
        
        if agent_object is not None:
            # Use provided agent object
            dqn_agent = VisualGameCompatibleDQN(agent_object)
            self.game.set_dqn_agent(dqn_agent)
            print("✅ Loaded DQN agent from provided object")
            return True
            
        elif agent_file_path is not None:
            try:
                # Load from file
                with open(agent_file_path, 'rb') as f:
                    agent_data = pickle.load(f)
                
                # Create compatible agent (this would need adaptation based on your specific agent format)
                dqn_agent = VisualGameCompatibleDQN()
                # You'd need to reconstruct the agent from saved data here
                print(f"✅ Loaded DQN agent from {agent_file_path}")
                self.game.set_dqn_agent(dqn_agent)
                return True
                
            except Exception as e:
                print(f"❌ Failed to load agent from {agent_file_path}: {e}")
                return False
        else:
            # Create demo agent
            dqn_agent = VisualGameCompatibleDQN()  # No trained agent
            self.game.set_dqn_agent(dqn_agent)
            print("⚠️ Using demo rule-based agent")
            return True
    
    def setup_game_objects(self):
        """Initialize game objects for drawing"""
        self.ax.clear()
        self.ax.set_xlim(0, self.game.width)
        self.ax.set_ylim(0, self.game.height)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('black')
        
        # Create paddle
        self.paddle_rect = Rectangle(
            (self.game.paddle_x, self.game.paddle_y),
            self.game.paddle_width, self.game.paddle_height,
            facecolor=self.game.colors['paddle']
        )
        self.ax.add_patch(self.paddle_rect)
        
        # Create ball
        self.ball_circle = Circle(
            (self.game.ball_x, self.game.ball_y),
            self.game.ball_radius,
            facecolor=self.game.colors['ball']
        )
        self.ax.add_patch(self.ball_circle)
        
        # Create bricks
        self.brick_rects = []
        for brick in self.game.bricks:
            if brick['active']:
                rect = Rectangle(
                    (brick['x'], brick['y']),
                    self.game.brick_width, self.game.brick_height,
                    facecolor=brick['color'],
                    edgecolor='white',
                    linewidth=1
                )
                self.ax.add_patch(rect)
                self.brick_rects.append(rect)
        
        self.update_text_display()
    
    def update_text_display(self):
        """Update text display"""
        # Clear previous text
        for text_obj in self.text_objects:
            text_obj.remove()
        self.text_objects.clear()
        
        # Game stats
        stats = self.game.game_stats
        agent_type = "DQN AI" if (self.game.dqn_agent and self.game.dqn_agent.is_trained) else "Rule-based AI"
        mode_text = f"{agent_type}" if self.game.ai_mode else "HUMAN"
        
        text_info = [
            f"Score: {stats['score']}",
            f"Lives: {stats['lives']}",
            f"Paddle Hits: {stats['paddle_hits']}",
            f"Bricks: {stats['bricks_destroyed']}/{len(self.game.bricks)}",
            f"Mode: {mode_text}",
            f"Frame: {self.frame_count}"
        ]
        
        # Add performance stats for AI
        if self.game.ai_mode and len(self.performance_history) > 0:
            text_info.append(f"Avg Score: {np.mean(self.performance_history):.1f}")
        
        for i, text in enumerate(text_info):
            text_obj = self.ax.text(
                10, self.game.height - 30 - (i * 25), text,
                color=self.game.colors['text'],
                fontsize=12,
                fontweight='bold'
            )
            self.text_objects.append(text_obj)
        
        # Game over/won messages
        if self.game.game_over:
            game_over_text = self.ax.text(
                self.game.width // 2, self.game.height // 2,
                "GAME OVER\nPress 'R' to restart",
                color='red', fontsize=24, fontweight='bold',
                ha='center', va='center'
            )
            self.text_objects.append(game_over_text)
        elif self.game.game_won:
            win_text = self.ax.text(
                self.game.width // 2, self.game.height // 2,
                "YOU WON!\nPress 'R' to restart",
                color='green', fontsize=24, fontweight='bold',
                ha='center', va='center'
            )
            self.text_objects.append(win_text)
    
    def update_frame(self, frame):
        """Animation update function"""
        if not self.is_running:
            return []
        
        self.frame_count += 1
        
        # Update game
        self.game.update_game()
        
        # Track AI actions
        if self.game.ai_mode and self.game.dqn_agent:
            state = self.game.get_state_for_dqn()
            action = self.game.dqn_agent.act(state)
            self.action_history.append(action)
        
        # Update visual objects
        self.paddle_rect.set_x(self.game.paddle_x)
        self.ball_circle.center = (self.game.ball_x, self.game.ball_y)
        
        # Update bricks
        for i, brick in enumerate(self.game.bricks):
            if not brick['active'] and i < len(self.brick_rects):
                self.brick_rects[i].remove()
        
        # Rebuild active brick list
        active_rects = []
        for i, brick in enumerate(self.game.bricks):
            if brick['active'] and i < len(self.brick_rects):
                active_rects.append(self.brick_rects[i])
        self.brick_rects = active_rects
        
        self.update_text_display()
        
        # Track performance
        if self.game.ai_mode:
            self.performance_history.append(self.game.game_stats['score'])
        
        # Auto-restart for AI demo
        if (self.game.game_over or self.game.game_won) and self.game.ai_mode:
            if self.frame_count % 120 == 0:  # Wait 2 seconds
                self.restart_game()
        
        return [self.paddle_rect, self.ball_circle] + self.brick_rects + self.text_objects
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == 'left' or event.key == 'a':
            if not self.game.ai_mode and self.game.paddle_x > 0:
                self.game.paddle_x -= 15
        elif event.key == 'right' or event.key == 'd':
            if not self.game.ai_mode and self.game.paddle_x < self.game.width - self.game.paddle_width:
                self.game.paddle_x += 15
        elif event.key == 'r':
            self.restart_game()
        elif event.key == 't':
            self.toggle_ai_mode()
        elif event.key == 'q':
            self.stop_game()
        elif event.key == 's':
            self.show_ai_stats()
    
    def toggle_ai_mode(self):
        """Toggle between AI and human control"""
        self.game.ai_mode = not self.game.ai_mode
        mode = "AI" if self.game.ai_mode else "Human"
        agent_type = "DQN" if (self.game.dqn_agent and self.game.dqn_agent.is_trained) else "Rule-based"
        print(f"Switched to {mode} mode ({agent_type})")
    
    def restart_game(self):
        """Restart the game"""
        self.game.reset_game()
        self.frame_count = 0
        self.setup_game_objects()
        print("Game restarted!")
    
    def show_ai_stats(self):
        """Show AI performance statistics"""
        if len(self.action_history) > 0:
            actions = list(self.action_history)
            action_counts = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
            total_actions = len(actions)
            
            print("\n📊 AI PERFORMANCE STATS")
            print("-" * 30)
            print(f"Total Actions: {total_actions}")
            print(f"No-op: {action_counts[0]} ({action_counts[0]/total_actions:.1%})")
            print(f"Left: {action_counts[1]} ({action_counts[1]/total_actions:.1%})")
            print(f"Right: {action_counts[2]} ({action_counts[2]/total_actions:.1%})")
            
            if len(self.performance_history) > 0:
                print(f"Average Score: {np.mean(self.performance_history):.1f}")
                print(f"Best Score: {np.max(self.performance_history)}")
                print(f"Games Played: {len(self.performance_history)}")
    
    def start_game(self):
        """Start the game"""
        self.is_running = True
        self.setup_game_objects()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_frame, interval=50, blit=False, repeat=True
        )
        
        # Show instructions
        print("\n" + "="*60)
        print("🎮 DQN VISUAL BREAKOUT GAME")
        print("="*60)
        print("Controls:")
        print("  Human Mode: Arrow keys or A/D to move paddle")
        print("  'T' - Toggle AI/Human mode")
        print("  'R' - Restart game")
        print("  'S' - Show AI statistics")
        print("  'Q' - Quit game")
        print("")
        agent_status = "✅ DQN Agent Loaded" if (self.game.dqn_agent and self.game.dqn_agent.is_trained) else "⚠️ Demo Agent Only"
        print(f"Agent Status: {agent_status}")
        print("Press 'T' to switch to AI mode!")
        print("="*60)
        
        plt.show()
    
    def stop_game(self):
        """Stop the game"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close()
        print("Game stopped!")

# MAIN INTEGRATION FUNCTION
def integrate_dqn_with_visual_game(trained_agent=None, agent_file_path=None):
    """
    Main function to integrate your trained DQN agent with the visual game
    
    Parameters:
    - trained_agent: Your trained agent object from comprehensive_training()
    - agent_file_path: Path to saved agent file (.pkl)
    """
    
    print("🚀 INTEGRATING DQN AGENT WITH VISUAL GAME")
    print("=" * 50)
    
    # Create game interface
    game_interface = DQNGameInterface()
    
    # Load the DQN agent
    success = game_interface.load_trained_dqn_agent(agent_file_path, trained_agent)
    
    if success:
        # Start in AI mode to show the DQN in action
        game_interface.toggle_ai_mode()
        print("Starting in AI mode to demonstrate DQN behavior")
        
        # Start the game
        game_interface.start_game()
    else:
        print("❌ Failed to load DQN agent")

# STEP-BY-STEP INTEGRATION EXAMPLE
def step_by_step_integration_example():
    """
    Complete example showing how to train a DQN and then visualize it
    """
    
    print("📚 STEP-BY-STEP DQN INTEGRATION EXAMPLE")
    print("=" * 50)
    
    print("\nStep 1: Train DQN Agent (simplified version)")
    print("-" * 30)
    
    # This is a simplified training - you'd use your comprehensive_training() function
    # For demo purposes, we'll create a mock trained agent
    
    class MockTrainedAgent:
        def __init__(self):
            self.q_network = None  # Would be your actual network
            self.action_size = 3
            self.epsilon = 0.01
            
        def act(self, state, record_strategy=False):
            # Simple mock behavior
            paddle_x = state[0] if len(state) > 0 else 0.5
            ball_x = state[1] if len(state) > 1 else 0.5
            
            if ball_x < paddle_x - 0.1:
                return 1  # Move left
            elif ball_x > paddle_x + 0.1:
                return 2  # Move right
            else:
                return 0  # Stay
    
    # Create mock agent
    mock_agent = MockTrainedAgent()
    print("✅ Mock agent created (replace with your trained agent)")
    
    print("\nStep 2: Integrate with Visual Game")
    print("-" * 30)
    
    # Integrate with visual game
    integrate_dqn_with_visual_game(trained_agent=mock_agent)

# REAL INTEGRATION WITH YOUR TRAINED AGENT
def integrate_your_trained_agent():
    """
    Use this function to integrate your actual trained agent
    """
    
    print("🎯 INTEGRATING YOUR TRAINED AGENT")
    print("=" * 40)
    print("To integrate your trained DQN agent:")
    print("")
    print("1. First, train your agent:")
    print("   agent, training_data = comprehensive_training(episodes=1000)")
    print("")
    print("2. Then integrate with visual game:")
    print("   integrate_dqn_with_visual_game(trained_agent=agent)")
    print("")
    print("3. Or load from saved file:")
    print("   integrate_dqn_with_visual_game(agent_file_path='adaptive_breakout_agent.pkl')")
    print("")
    print("Running demo with mock agent...")
    
    # For now, run the demo
    step_by_step_integration_example()

# Main execution
if __name__ == "__main__":
    print("🎮 DQN VISUAL GAME INTEGRATION")
    print("=" * 50)
    print("Choose an option:")
    print("1. Demo integration (mock agent)")
    print("2. Integration guide")
    print("3. Load from file (if you have a saved agent)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        step_by_step_integration_example()
    elif choice == "2":
        integrate_your_trained_agent()
    elif choice == "3":
        file_path = input("Enter path to saved agent file: ").strip()
        if file_path:
            integrate_dqn_with_visual_game(agent_file_path=file_path)
        else:
            print("No file path provided, running demo...")
            step_by_step_integration_example()
    else:
        print("Invalid choice, running demo...")
        step_by_step_integration_example()