"""
Reinforcement Learning Agent for Snake Game Optimization
Uses Q-Learning to train an agent to play the Snake game
"""

import numpy as np
from collections import deque
import json
import os

class SnakeGameEnvironment:
    """Snake game environment for RL training"""
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Snake starts in the middle
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.food = self._spawn_food()
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        return self._get_state()
    
    def _spawn_food(self):
        """Spawn food at random position not occupied by snake"""
        while True:
            food = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        """Get current game state as a tuple"""
        head = self.snake[0]
        food = self.food
        
        # Simplified state: relative positions and direction
        state = (
            head[0], head[1],
            food[0], food[1],
            len(self.snake)
        )
        return state
    
    def step(self, action):
        """
        Execute one action
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        """
        head_x, head_y = self.snake[0]
        
        # Move based on action
        if action == 0:  # Up
            new_head = (head_x - 1, head_y)
        elif action == 1:  # Down
            new_head = (head_x + 1, head_y)
        elif action == 2:  # Left
            new_head = (head_x, head_y - 1)
        else:  # Right
            new_head = (head_x, head_y + 1)
        
        reward = -0.01  # Small penalty for each step
        done = False
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or 
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = -1
            done = True
            return self._get_state(), reward, done
        
        # Check self collision
        if new_head in self.snake:
            reward = -1
            done = True
            return self._get_state(), reward, done
        
        # Move snake
        self.snake.appendleft(new_head)
        
        # Check food collision
        if new_head == self.food:
            reward = 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_state(), reward, done


class QLearningAgent:
    """Q-Learning agent for Snake game"""
    
    def __init__(self, grid_size=10, learning_rate=0.1, discount_factor=0.95):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = [0, 1, 2, 3]  # Up, Down, Left, Right
        self.q_table = {}
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete"""
        return tuple(state)
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        state = self._discretize_state(state)
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state].get(action, 0.0)
    
    def choose_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        state = self._discretize_state(state)
        
        if training and np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a in self.actions if q_values[a] == max_q]
        return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_q_table(self, filename='q_table.npy'):
        """Save Q-table to numpy file"""
        # Convert dict to numpy-compatible format
        q_data = {str(k): v for k, v in self.q_table.items()}
        np.save(filename, q_data, allow_pickle=True)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename='q_table.npy'):
        """Load Q-table from numpy file"""
        q_data = np.load(filename, allow_pickle=True).item()
        self.q_table = {eval(k): v for k, v in q_data.items()}
        print(f"Q-table loaded from {filename}")


def train_agent(episodes=1000, grid_size=10):
    """Train the RL agent"""
    env = SnakeGameEnvironment(grid_size=grid_size)
    agent = QLearningAgent(grid_size=grid_size)
    
    scores = []
    
    print("Training RL Agent on Snake Game...")
    print(f"Episodes: {episodes}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            
            agent.update_q_value(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        agent.decay_epsilon()
        scores.append(len(env.snake))
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode + 1}/{episodes} | Avg Score (last 100): {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Save the trained Q-table
    agent.save_q_table('q_table.npy')
    
    return agent, scores


def evaluate_agent(agent, episodes=10, grid_size=10):
    """Evaluate the trained agent"""
    env = SnakeGameEnvironment(grid_size=grid_size)
    scores = []
    
    print("\nEvaluating Trained Agent...")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            next_state, reward, done = env.step(action)
            state = next_state
        
        scores.append(len(env.snake))
        print(f"Evaluation {episode + 1}/{episodes} | Score: {len(env.snake)}")
    
    print(f"\nAverage Evaluation Score: {np.mean(scores):.2f}")
    return scores


if __name__ == "__main__":
    # Train the agent
    agent, training_scores = train_agent(episodes=500, grid_size=10)
    
    # Evaluate the trained agent
    eval_scores = evaluate_agent(agent, episodes=5, grid_size=10)
