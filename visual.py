import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import random
import time
from collections import deque
import pickle

class VisualGameCompatibleDQN:
    """DQN Agent adapted to work with the visual game interface"""
    
    def __init__(self, trained_agent=None):
        if trained_agent is not None:
            self.q_network = trained_agent.q_network
            self.epsilon = 0.01
            self.action_size = trained_agent.action_size
            self.is_trained = True
            print("Loaded pre-trained DQN agent")
        else:
            self.q_network = None
            self.epsilon = 0.1
            self.action_size = 3
            self.is_trained = False
            print("Using simple rule-based agent")
    
    def convert_visual_state_to_dqn_format(self, visual_game_state):
        """Convert the visual game state to DQN-compatible format"""
        
        if not self.is_trained:
            return visual_game_state
        
        # Create simplified grid representation
        grid_size = 84
        state_grid = np.zeros((grid_size, grid_size))
        
        # Normalize coordinates
        paddle_x = int(visual_game_state[0] * grid_size)
        ball_x = int(visual_game_state[1] * grid_size)
        ball_y = int(visual_game_state[2] * grid_size)
        
        # Ensure coordinates are within bounds
        paddle_x = max(0, min(grid_size-1, paddle_x))
        ball_x = max(0, min(grid_size-1, ball_x))
        ball_y = max(0, min(grid_size-1, ball_y))
        
        # Place paddle in grid
        paddle_y = grid_size - 5
        for i in range(max(0, paddle_x-5), min(grid_size, paddle_x+5)):
            for j in range(max(0, paddle_y-2), min(grid_size, paddle_y+2)):
                state_grid[j, i] = 1.0
        
        # Place ball in grid
        for i in range(max(0, ball_x-1), min(grid_size, ball_x+1)):
            for j in range(max(0, ball_y-1), min(grid_size, ball_y+1)):
                state_grid[j, i] = 2.0
        
        # Add brick information
        brick_density = visual_game_state[5]
        for i in range(0, grid_size, 8):
            for j in range(0, 20):
                if random.random() < brick_density:
                    state_grid[j, i] = 3.0
        
        return state_grid.flatten()
    
    def act(self, visual_game_state, record_strategy=False):
        """Get action from DQN agent given visual game state"""
        
        if not self.is_trained or self.q_network is None:
            return self.simple_rule_based_action(visual_game_state)
        
        dqn_state = self.convert_visual_state_to_dqn_format(visual_game_state)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
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
        if ball_dy > 0:
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

class DifficultyLevels:
    """Defines different difficulty levels with varying adaptive parameters"""
    
    LEVELS = {
        'EASY': {
            'name': 'Easy Mode',
            'description': 'Minimal adaptive changes',
            'color': '#00FF00',
            'paddle_speed_change_freq': 200,
            'paddle_speed_range': (0.8, 1.2),
            'ball_speed_increase_chance': 0.0005,
            'ball_speed_increase_amount': 0.1,
            'ball_speed_max': 1.5,
            'paddle_size_change_freq': 1000,
            'paddle_size_range': (0.9, 1.1),
            'brick_regen_multiplier': 0.5,
            'brick_regen_max': 0.005,
        },
        'NORMAL': {
            'name': 'Normal Mode',
            'description': 'Standard adaptive difficulty',
            'color': '#FFFF00',
            'paddle_speed_change_freq': 100,
            'paddle_speed_range': (0.5, 1.5),
            'ball_speed_increase_chance': 0.001,
            'ball_speed_increase_amount': 0.2,
            'ball_speed_max': 2.0,
            'paddle_size_change_freq': 500,
            'paddle_size_range': (0.7, 1.3),
            'brick_regen_multiplier': 1.0,
            'brick_regen_max': 0.01,
        },
        'HARD': {
            'name': 'Hard Mode',
            'description': 'Aggressive adaptive changes',
            'color': '#FF8800',
            'paddle_speed_change_freq': 75,
            'paddle_speed_range': (0.3, 1.8),
            'ball_speed_increase_chance': 0.002,
            'ball_speed_increase_amount': 0.3,
            'ball_speed_max': 2.5,
            'paddle_size_change_freq': 300,
            'paddle_size_range': (0.5, 1.5),
            'brick_regen_multiplier': 1.5,
            'brick_regen_max': 0.02,
        },
        'EXTREME': {
            'name': 'Extreme Mode',
            'description': 'Maximum chaos',
            'color': '#FF0000',
            'paddle_speed_change_freq': 50,
            'paddle_speed_range': (0.2, 2.0),
            'ball_speed_increase_chance': 0.003,
            'ball_speed_increase_amount': 0.4,
            'ball_speed_max': 3.0,
            'paddle_size_change_freq': 200,
            'paddle_size_range': (0.4, 1.8),
            'brick_regen_multiplier': 2.0,
            'brick_regen_max': 0.03,
        }
    }
    
    @classmethod
    def get_level_names(cls):
        return list(cls.LEVELS.keys())
    
    @classmethod
    def get_level_config(cls, level_name):
        return cls.LEVELS.get(level_name.upper(), cls.LEVELS['NORMAL'])

class DQNVisualBreakout:
    def __init__(self, width=800, height=600, difficulty_level='NORMAL'):
        self.width = width
        self.height = height
        
        # Set difficulty level
        self.difficulty_level = difficulty_level.upper()
        self.difficulty_config = DifficultyLevels.get_level_config(self.difficulty_level)
        
        # Base game object dimensions
        self.base_paddle_width = 80
        self.paddle_height = 15
        self.ball_radius = 8
        self.brick_width = 75
        self.brick_height = 20
        
        # Base speeds
        self.base_paddle_speed = 8
        self.base_ball_speed = 4
        
        # Current modifiers
        self.paddle_speed_modifier = 1.0
        self.ball_speed_modifier = 1.0
        self.paddle_width_modifier = 1.0
        
        # Current actual values
        self.paddle_width = self.base_paddle_width
        self.current_paddle_speed = self.base_paddle_speed
        self.current_ball_speed = self.base_ball_speed
        
        # Frame tracking
        self.frame_count = 0
        self.difficulty_changes = []
        self.brick_regen_prob = 0.0
        
        # Colors
        self.colors = {
            'background': '#000000',
            'paddle': '#FFFFFF',
            'ball': '#FF6B6B',
            'bricks': ['#FF9F43', '#10AC84', '#5F27CD', '#00D2D3', '#FF3838', '#2E86AB'],
            'text': '#FFFFFF',
            'difficulty_text': self.difficulty_config['color'],
            'level_indicator': self.difficulty_config['color']
        }
        
        # Game state
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
        
        # Notifications
        self.difficulty_notifications = deque(maxlen=5)
        
        self.reset_game()
        
    def set_difficulty_level(self, level_name):
        """Change difficulty level during gameplay"""
        old_level = self.difficulty_level
        self.difficulty_level = level_name.upper()
        self.difficulty_config = DifficultyLevels.get_level_config(self.difficulty_level)
        
        # Update colors
        self.colors['difficulty_text'] = self.difficulty_config['color']
        self.colors['level_indicator'] = self.difficulty_config['color']
        
        # Add notification
        self.difficulty_notifications.append({
            'message': f"Difficulty: {self.difficulty_config['name']}",
            'frames_left': 240,
            'color': self.difficulty_config['color']
        })
        
        print(f"Difficulty changed to {self.difficulty_level}")
        return True
    
    def reset_game(self):
        """Reset the game to initial state"""
        # Paddle position
        self.paddle_x = self.width // 2 - 40
        self.paddle_y = self.height - 50
        
        # Ball position and velocity
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = random.choice([-self.current_ball_speed, self.current_ball_speed])
        self.ball_dy = -self.current_ball_speed
        
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
        
        # Reset difficulty modifiers
        self.paddle_speed_modifier = 1.0
        self.ball_speed_modifier = 1.0
        self.paddle_width_modifier = 1.0
        self.frame_count = 0
        self.brick_regen_prob = 0.0
        self.difficulty_changes.clear()
        
        self.update_current_values()
        
        self.game_over = False
        self.game_won = False
    
    def update_current_values(self):
        """Update current game values based on modifiers"""
        self.paddle_width = int(self.base_paddle_width * self.paddle_width_modifier)
        self.current_paddle_speed = int(self.base_paddle_speed * self.paddle_speed_modifier)
        self.current_ball_speed = self.base_ball_speed * self.ball_speed_modifier
    
    def apply_difficulty_changes(self):
        """Apply dynamic difficulty changes based on selected difficulty level"""
        self.frame_count += 1
        config = self.difficulty_config
        
        # Paddle speed variation
        if self.frame_count % config['paddle_speed_change_freq'] == 0:
            old_modifier = self.paddle_speed_modifier
            min_speed, max_speed = config['paddle_speed_range']
            self.paddle_speed_modifier = random.uniform(min_speed, max_speed)
            
            change_info = {
                'type': 'paddle_speed',
                'frame': self.frame_count,
                'old_value': old_modifier,
                'new_value': self.paddle_speed_modifier,
                'magnitude': abs(self.paddle_speed_modifier - old_modifier)
            }
            self.difficulty_changes.append(change_info)
            
            self.difficulty_notifications.append({
                'message': f"Paddle Speed: {self.paddle_speed_modifier:.2f}x",
                'frames_left': 180,
                'color': config['color']
            })
            
            self.update_current_values()
        
        # Ball speed increase
        if random.random() < config['ball_speed_increase_chance']:
            old_modifier = self.ball_speed_modifier
            self.ball_speed_modifier = min(
                config['ball_speed_max'], 
                self.ball_speed_modifier + config['ball_speed_increase_amount']
            )
            
            if self.ball_speed_modifier != old_modifier:
                change_info = {
                    'type': 'ball_speed',
                    'frame': self.frame_count,
                    'old_value': old_modifier,
                    'new_value': self.ball_speed_modifier,
                    'magnitude': abs(self.ball_speed_modifier - old_modifier)
                }
                self.difficulty_changes.append(change_info)
                
                self.difficulty_notifications.append({
                    'message': f"Ball Speed: {self.ball_speed_modifier:.2f}x",
                    'frames_left': 180,
                    'color': '#FF6B6B'
                })
                
                # Update ball velocity immediately
                speed_ratio = self.ball_speed_modifier / old_modifier
                self.ball_dx *= speed_ratio
                self.ball_dy *= speed_ratio
        
        # Paddle size change
        if self.frame_count % config['paddle_size_change_freq'] == 0:
            old_modifier = self.paddle_width_modifier
            min_size, max_size = config['paddle_size_range']
            self.paddle_width_modifier = random.uniform(min_size, max_size)
            
            change_info = {
                'type': 'paddle_width',
                'frame': self.frame_count,
                'old_value': old_modifier,
                'new_value': self.paddle_width_modifier,
                'magnitude': abs(self.paddle_width_modifier - old_modifier)
            }
            self.difficulty_changes.append(change_info)
            
            self.difficulty_notifications.append({
                'message': f"Paddle Size: {self.paddle_width_modifier:.2f}x",
                'frames_left': 180,
                'color': '#FFFFFF'
            })
            
            self.update_current_values()
        
        # Brick regeneration
        old_prob = self.brick_regen_prob
        base_prob = (self.frame_count / 100000) * config['brick_regen_multiplier']
        self.brick_regen_prob = min(config['brick_regen_max'], base_prob)
        
        # Actual brick regeneration
        if random.random() < self.brick_regen_prob:
            destroyed_bricks = [i for i, brick in enumerate(self.bricks) if not brick['active']]
            if destroyed_bricks:
                regen_idx = random.choice(destroyed_bricks)
                self.bricks[regen_idx]['active'] = True
                
                self.difficulty_notifications.append({
                    'message': "Brick Regenerated!",
                    'frames_left': 120,
                    'color': '#10AC84'
                })
        
        # Update notification timers
        for notification in list(self.difficulty_notifications):
            notification['frames_left'] -= 1
            if notification['frames_left'] <= 0:
                self.difficulty_notifications.remove(notification)
    
    def get_state_for_dqn(self):
        """Get game state in format suitable for DQN"""
        state = np.array([
            self.paddle_x / self.width,
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_dx / 10.0,
            self.ball_dy / 10.0,
            len([b for b in self.bricks if b['active']]) / len(self.bricks)
        ])
        return state
    
    def set_dqn_agent(self, dqn_agent):
        """Set the DQN agent"""
        self.dqn_agent = dqn_agent
        agent_type = 'Trained' if dqn_agent.is_trained else 'Rule-based'
        print(f"DQN Agent set: {agent_type}")
    
    def update_game(self, action=None):
        """Update game state with adaptive difficulty"""
        if self.game_over or self.game_won:
            return
        
        self.apply_difficulty_changes()
        
        # Get action from DQN if in AI mode
        if self.ai_mode and self.dqn_agent is not None and action is None:
            state = self.get_state_for_dqn()
            action = self.dqn_agent.act(state)
        
        # Apply action with current paddle speed
        if action == 1 and self.paddle_x > 0:
            self.paddle_x -= self.current_paddle_speed
            self.paddle_x = max(0, self.paddle_x)
        elif action == 2 and self.paddle_x < self.width - self.paddle_width:
            self.paddle_x += self.current_paddle_speed
            self.paddle_x = min(self.width - self.paddle_width, self.paddle_x)
        
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
            max_speed = 8 * self.ball_speed_modifier
            speed = np.sqrt(self.ball_dx**2 + self.ball_dy**2)
            if speed > max_speed:
                self.ball_dx = (self.ball_dx / speed) * max_speed
                self.ball_dy = (self.ball_dy / speed) * max_speed
            
            self.game_stats['paddle_hits'] += 1
            self.game_stats['score'] += 1
        
        # Store old ball position before update
        prev_ball_x = self.ball_x - self.ball_dx
        prev_ball_y = self.ball_y - self.ball_dy

        # Ball collision with bricks
        for brick in self.bricks:
            if not brick['active']:
                continue

            if (self.ball_x + self.ball_radius >= brick['x'] and
                self.ball_x - self.ball_radius <= brick['x'] + self.brick_width and
                self.ball_y + self.ball_radius >= brick['y'] and
                self.ball_y - self.ball_radius <= brick['y'] + self.brick_height):

                # Determine where the ball came from
                from_left = prev_ball_x + self.ball_radius <= brick['x']
                from_right = prev_ball_x - self.ball_radius >= brick['x'] + self.brick_width
                from_top = prev_ball_y + self.ball_radius <= brick['y']
                from_bottom = prev_ball_y - self.ball_radius >= brick['y'] + self.brick_height

                # Flip velocity based on where it came from
                if from_left or from_right:
                    self.ball_dx = -self.ball_dx
                elif from_top or from_bottom:
                    self.ball_dy = -self.ball_dy
                else:
                    # Fallback if unclear (corner hit)
                    self.ball_dy = -self.ball_dy

                brick['active'] = False
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
                self.ball_dx = random.choice([-self.current_ball_speed, self.current_ball_speed])
                self.ball_dy = -self.current_ball_speed
        
        # Check win condition
        if all(not brick['active'] for brick in self.bricks):
            self.game_won = True
        
        self.game_stats['game_time'] += 1

class DQNGameInterface:
    def __init__(self, difficulty_level='NORMAL'):
        self.current_difficulty = difficulty_level
        self.game = DQNVisualBreakout(difficulty_level=difficulty_level)
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
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
        
    def change_difficulty(self, new_difficulty):
        """Change difficulty level during gameplay"""
        if new_difficulty.upper() in DifficultyLevels.get_level_names():
            self.current_difficulty = new_difficulty.upper()
            self.game.set_difficulty_level(new_difficulty)
            return True
        return False
        
    def load_trained_dqn_agent(self, agent_file_path=None, agent_object=None):
        """Load a trained DQN agent"""
        
        if agent_object is not None:
            dqn_agent = VisualGameCompatibleDQN(agent_object)
            self.game.set_dqn_agent(dqn_agent)
            print("Loaded DQN agent from provided object")
            return True
            
        elif agent_file_path is not None:
            try:
                with open(agent_file_path, 'rb') as f:
                    agent_data = pickle.load(f)
                
                dqn_agent = VisualGameCompatibleDQN()
                print(f"Loaded DQN agent from {agent_file_path}")
                self.game.set_dqn_agent(dqn_agent)
                return True
                
            except Exception as e:
                print(f"Failed to load agent from {agent_file_path}: {e}")
                return False
        else:
            dqn_agent = VisualGameCompatibleDQN()
            self.game.set_dqn_agent(dqn_agent)
            print("Using demo rule-based agent")
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
        """Update text display with difficulty information"""
        # Clear previous text
        for text_obj in self.text_objects:
            text_obj.remove()
        self.text_objects.clear()
        
        # Game stats
        stats = self.game.game_stats
        agent_type = "DQN AI" if (self.game.dqn_agent and self.game.dqn_agent.is_trained) else "Rule-based AI"
        mode_text = f"{agent_type}" if self.game.ai_mode else "HUMAN"
        
        # Basic game info
        text_info = [
            f"Score: {stats['score']}",
            f"Lives: {stats['lives']}",
            f"Paddle Hits: {stats['paddle_hits']}",
            f"Bricks: {stats['bricks_destroyed']}/{len(self.game.bricks)}",
            f"Mode: {mode_text}",
            f"Frame: {self.frame_count}"
        ]
        
        # Difficulty level info
        config = self.game.difficulty_config
        text_info.extend([
            "",
            f"Difficulty: {config['name']}",
            f"{config['description']}"
        ])
        
        # Current modifiers
        text_info.extend([
            "",
            "Current Modifiers:",
            f"Paddle Speed: {self.game.paddle_speed_modifier:.2f}x",
            f"Ball Speed: {self.game.ball_speed_modifier:.2f}x", 
            f"Paddle Size: {self.game.paddle_width_modifier:.2f}x",
            f"Brick Regen: {self.game.brick_regen_prob:.4f}",
            f"Changes: {len(self.game.difficulty_changes)}"
        ])
        
        # Add performance stats for AI
        if self.game.ai_mode and len(self.performance_history) > 0:
            text_info.extend([
                "",
                f"Avg Score: {np.mean(self.performance_history):.1f}",
                f"Best Score: {np.max(self.performance_history)}"
            ])
        
        # Display main text
        for i, text in enumerate(text_info):
            if text.startswith("Difficulty:"):
                color = config['color']
                weight = 'bold'
            elif text.startswith("Current Modifiers:"):
                color = self.game.colors['difficulty_text']
                weight = 'bold'
            else:
                color = self.game.colors['text']
                weight = 'normal'
                
            text_obj = self.ax.text(
                10, self.game.height - 25 - (i * 18), text,
                color=color,
                fontsize=9,
                fontweight=weight
            )
            self.text_objects.append(text_obj)
        
        # Difficulty level indicator
        level_text = self.ax.text(
            self.game.width - 10, self.game.height - 30,
            f"{config['name'].upper()}",
            color=config['color'],
            fontsize=16,
            fontweight='bold',
            ha='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8, edgecolor=config['color'])
        )
        self.text_objects.append(level_text)
        
        # Difficulty notifications
        for i, notification in enumerate(self.game.difficulty_notifications):
            alpha = min(1.0, notification['frames_left'] / 60.0)
            text_obj = self.ax.text(
                self.game.width - 250, self.game.height - 80 - (i * 30),
                notification['message'],
                color=notification['color'],
                fontsize=11,
                fontweight='bold',
                alpha=alpha,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
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
        
        self.game.update_game()
        
        # Track AI actions
        if self.game.ai_mode and self.game.dqn_agent:
            state = self.game.get_state_for_dqn()
            action = self.game.dqn_agent.act(state)
            self.action_history.append(action)
        
        # Update visual objects
        self.paddle_rect.set_x(self.game.paddle_x)
        self.paddle_rect.set_width(self.game.paddle_width)
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
            if self.frame_count % 120 == 0:
                self.restart_game()
        
        return [self.paddle_rect, self.ball_circle] + self.brick_rects + self.text_objects
    
    def on_key_press(self, event):
        """Handle keyboard input including difficulty changes"""
        if event.key == 'left' or event.key == 'a':
            if not self.game.ai_mode and self.game.paddle_x > 0:
                self.game.paddle_x -= self.game.current_paddle_speed
                self.game.paddle_x = max(0, self.game.paddle_x)
        elif event.key == 'right' or event.key == 'd':
            if not self.game.ai_mode and self.game.paddle_x < self.game.width - self.game.paddle_width:
                self.game.paddle_x += self.game.current_paddle_speed
                self.game.paddle_x = min(self.game.width - self.game.paddle_width, self.game.paddle_x)
        elif event.key == 'r':
            self.restart_game()
        elif event.key == 't':
            self.toggle_ai_mode()
        elif event.key == 'q':
            self.stop_game()
        elif event.key == 's':
            self.show_ai_stats()
        elif event.key == 'i':
            self.show_difficulty_info()
        # Difficulty level changes
        elif event.key == '1':
            self.change_difficulty('EASY')
        elif event.key == '2':
            self.change_difficulty('NORMAL')
        elif event.key == '3':
            self.change_difficulty('HARD')
        elif event.key == '4':
            self.change_difficulty('EXTREME')
        elif event.key == 'c':
            self.show_difficulty_menu()
    
    def show_difficulty_menu(self):
        """Show difficulty selection menu"""
        print("\nDifficulty Levels:")
        for i, (level_name, config) in enumerate(DifficultyLevels.LEVELS.items(), 1):
            current = " (current)" if level_name == self.current_difficulty else ""
            print(f"{i}. {config['name']}{current} - {config['description']}")
        print("Press number keys 1-4 to change difficulty")
    
    def show_difficulty_info(self):
        """Show detailed difficulty information"""
        config = self.game.difficulty_config
        print(f"\nCurrent Difficulty: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Frame: {self.game.frame_count}")
        print(f"Paddle Speed Range: {config['paddle_speed_range'][0]:.1f}x - {config['paddle_speed_range'][1]:.1f}x")
        print(f"Ball Speed Max: {config['ball_speed_max']:.1f}x")
        print(f"Paddle Size Range: {config['paddle_size_range'][0]:.1f}x - {config['paddle_size_range'][1]:.1f}x")
        print(f"Brick Regeneration Max: {config['brick_regen_max']:.3%}")
        print(f"Total Changes: {len(self.game.difficulty_changes)}")
    
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
        print("Game restarted")
    
    def show_ai_stats(self):
        """Show AI performance statistics"""
        if len(self.action_history) > 0:
            actions = list(self.action_history)
            action_counts = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}
            total_actions = len(actions)
            
            print("\nAI Performance Stats:")
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
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.animation = animation.FuncAnimation(
            self.fig, self.update_frame, interval=50, blit=False, repeat=True
        )
        
        # Show instructions
        print("\nAdaptive DQN Visual Breakout")
        print("Difficulty Levels:")
        for i, (level_name, config) in enumerate(DifficultyLevels.LEVELS.items(), 1):
            current = " <- current" if level_name == self.current_difficulty else ""
            print(f"  {i}. {config['name']}: {config['description']}{current}")
        
        print("\nAdaptive Features:")
        print("  - Paddle speed varies randomly")
        print("  - Ball speed increases unpredictably")
        print("  - Paddle size changes periodically")
        print("  - Destroyed bricks can regenerate")
        
        print("\nControls:")
        print("  Arrow keys/A/D - Move paddle (human mode)")
        print("  T - Toggle AI/Human mode")
        print("  1-4 - Change difficulty")
        print("  C - Show difficulty menu")
        print("  R - Restart game")
        print("  S - Show AI stats")
        print("  I - Show difficulty info")
        print("  Q - Quit")
        
        agent_status = "DQN Agent Loaded" if (self.game.dqn_agent and self.game.dqn_agent.is_trained) else "Demo Agent Only"
        print(f"\nAgent Status: {agent_status}")
        print("Press T to switch to AI mode")
        
        plt.show()
    
    def stop_game(self):
        """Stop the game"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close()
        print("Game stopped")

def integrate_dqn_with_visual_game(trained_agent=None, agent_file_path=None, difficulty_level='NORMAL'):
    """
    Main function to integrate your trained DQN agent with the visual game
    
    Parameters:
    - trained_agent: Your trained agent object from comprehensive_training()
    - agent_file_path: Path to saved agent file (.pkl)
    - difficulty_level: Starting difficulty level ('EASY', 'NORMAL', 'HARD', 'EXTREME')
    """
    
    print("Integrating DQN agent with adaptive visual game")
    
    game_interface = DQNGameInterface(difficulty_level=difficulty_level)
    
    success = game_interface.load_trained_dqn_agent(agent_file_path, trained_agent)
    
    if success:
        game_interface.toggle_ai_mode()
        print(f"Starting in AI mode with {difficulty_level} difficulty")
        print("You can change difficulty during gameplay with number keys 1-4")
        
        game_interface.start_game()
    else:
        print("Failed to load DQN agent")

if __name__ == "__main__":
    print("Multi-Difficulty Adaptive DQN Visual Game")
    
    # Show difficulty options
    print("Available Difficulty Levels:")
    for i, (level_name, config) in enumerate(DifficultyLevels.LEVELS.items(), 1):
        print(f"  {i}. {config['name']}: {config['description']}")
    
    print("\nChoose an option:")
    print("1. Demo integration (rule-based agent)")
    print("2. Load from saved agent file")
    
    choice = input("Enter choice (1-2): ").strip()
    
    # Get difficulty level
    print("\nSelect starting difficulty:")
    print("1. Easy")
    print("2. Normal") 
    print("3. Hard")
    print("4. Extreme")
    
    diff_choice = input("Enter difficulty (1-4, default=2): ").strip()
    difficulty_map = {'1': 'EASY', '2': 'NORMAL', '3': 'HARD', '4': 'EXTREME'}
    difficulty = difficulty_map.get(diff_choice, 'NORMAL')
    
    if choice == "1":
        integrate_dqn_with_visual_game(difficulty_level=difficulty)
    elif choice == "2":
        file_path = input("Enter path to saved agent file: ").strip()
        if file_path:
            integrate_dqn_with_visual_game(agent_file_path=file_path, difficulty_level=difficulty)
        else:
            print("No file path provided, running demo...")
            integrate_dqn_with_visual_game(difficulty_level=difficulty)
    else:
        print("Invalid choice, running demo...")
        integrate_dqn_with_visual_game(difficulty_level=difficulty)
