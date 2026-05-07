"""
Visualization module for RL Snake Game training and evaluation
Displays training progress, Q-table statistics, and gameplay recording
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from rl_agent import SnakeGameEnvironment, QLearningAgent
import os


def plot_training_progress(scores, window_size=50):
    """
    Plot training progress with moving average
    
    Args:
        scores: List of scores from training
        window_size: Window size for moving average
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot raw scores
    ax1.plot(scores, alpha=0.6, label='Raw Score')
    
    # Plot moving average
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(scores)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Snake Length (Score)')
    ax1.set_title('Training Progress - Score over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot score distribution
    ax2.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Training progress plot saved as 'training_progress.png'")
    plt.close()


def plot_q_table_statistics(agent):
    """
    Plot Q-table statistics
    
    Args:
        agent: QLearningAgent instance
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract statistics
    state_count = len(agent.q_table)
    q_values_list = []
    for state_dict in agent.q_table.values():
        q_values_list.extend(state_dict.values())
    
    q_values_array = np.array(q_values_list)
    
    # Plot 1: Q-value distribution
    ax1.hist(q_values_array, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Q-value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Q-Value Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Q-value statistics
    stats_text = f"""
    Q-Table Statistics:
    
    States Explored: {state_count}
    Total Q-values: {len(q_values_list)}
    
    Q-Value Range:
    Min: {q_values_array.min():.4f}
    Max: {q_values_array.max():.4f}
    Mean: {q_values_array.mean():.4f}
    Std: {q_values_array.std():.4f}
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')
    ax2.set_title('Q-Table Statistics')
    
    # Plot 3: Action value comparison (average Q for each action)
    action_names = ['Up', 'Down', 'Left', 'Right']
    action_means = [[], [], [], []]
    
    for state_dict in agent.q_table.values():
        for action in range(4):
            if action in state_dict:
                action_means[action].append(state_dict[action])
    
    action_avgs = [np.mean(action_means[i]) if action_means[i] else 0 for i in range(4)]
    
    ax3.bar(action_names, action_avgs, color=['blue', 'red', 'green', 'orange'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Average Q-Value')
    ax3.set_title('Average Q-Value by Action')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Q-value statistics over time
    cumulative_states = []
    cumulative_avg_q = []
    
    states_processed = 0
    sum_q = 0
    for state_dict in agent.q_table.values():
        for q_value in state_dict.values():
            states_processed += 1
            sum_q += q_value
            cumulative_states.append(states_processed)
            cumulative_avg_q.append(sum_q / states_processed)
    
    if cumulative_states:
        ax4.plot(cumulative_states, cumulative_avg_q, linewidth=2, color='purple')
        ax4.set_xlabel('Q-values Processed')
        ax4.set_ylabel('Cumulative Average Q-Value')
        ax4.set_title('Learning Curve (Q-Value Convergence)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q_table_statistics.png', dpi=150, bbox_inches='tight')
    print("Q-table statistics plot saved as 'q_table_statistics.png'")
    plt.close()


def visualize_gameplay(agent, grid_size=10, num_games=3):
    """
    Visualize the trained agent playing
    
    Args:
        agent: QLearningAgent instance
        grid_size: Size of the game grid
        num_games: Number of games to visualize
    """
    fig, axes = plt.subplots(1, num_games, figsize=(15, 5))
    if num_games == 1:
        axes = [axes]
    
    for game_idx in range(num_games):
        env = SnakeGameEnvironment(grid_size=grid_size)
        state = env.reset()
        
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.choose_action(state, training=False)
            next_state, reward, done = env.step(action)
            
            if done:
                break
            
            state = next_state
            steps += 1
        
        # Visualize the final state
        ax = axes[game_idx]
        
        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(i, color='black', linewidth=0.5)
            ax.axvline(i, color='black', linewidth=0.5)
        
        # Draw snake
        for i, segment in enumerate(env.snake):
            color = 'darkgreen' if i == 0 else 'lightgreen'  # Head is darker
            rect = patches.Rectangle((segment[1], segment[0]), 1, 1, 
                                     linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
        
        # Draw food
        food_rect = patches.Rectangle((env.food[1], env.food[0]), 1, 1,
                                      linewidth=1, edgecolor='black', facecolor='red')
        ax.add_patch(food_rect)
        
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Game {game_idx + 1}\nSnake Length: {len(env.snake)}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('gameplay_visualization.png', dpi=150, bbox_inches='tight')
    print("Gameplay visualization saved as 'gameplay_visualization.png'")
    plt.close()


def create_summary_report(agent, training_scores, eval_scores):
    """
    Create a comprehensive summary report
    
    Args:
        agent: QLearningAgent instance
        training_scores: List of training scores
        eval_scores: List of evaluation scores
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    fig.suptitle('RL Snake Game - Training Summary Report', fontsize=16, fontweight='bold')
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training progress
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(training_scores, alpha=0.6, label='Training Score')
    moving_avg = np.convolve(training_scores, np.ones(50)/50, mode='valid')
    ax1.plot(range(49, len(training_scores)), moving_avg, 'r-', linewidth=2, label='50-Episode Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation results
    ax2 = fig.add_subplot(gs[1, 0])
    eval_episodes = range(1, len(eval_scores) + 1)
    ax2.bar(eval_episodes, eval_scores, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axhline(np.mean(eval_scores), color='red', linestyle='--', linewidth=2, label=f'Avg: {np.mean(eval_scores):.2f}')
    ax2.set_xlabel('Evaluation Episode')
    ax2.set_ylabel('Score')
    ax2.set_title('Evaluation Results')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Statistics summary
    ax3 = fig.add_subplot(gs[1, 1])
    stats_text = f"""
    TRAINING STATISTICS:
    
    Total Episodes: {len(training_scores)}
    Best Training Score: {max(training_scores)}
    Avg Training Score: {np.mean(training_scores):.2f}
    
    Q-TABLE STATISTICS:
    
    States Explored: {len(agent.q_table)}
    Final Epsilon: {agent.epsilon:.4f}
    Learning Rate: {agent.learning_rate}
    Discount Factor: {agent.discount_factor}
    
    EVALUATION STATISTICS:
    
    Evaluation Episodes: {len(eval_scores)}
    Best Eval Score: {max(eval_scores)}
    Avg Eval Score: {np.mean(eval_scores):.2f}
    Eval Std Dev: {np.std(eval_scores):.2f}
    """
    ax3.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax3.axis('off')
    
    # Plot 4: Score distribution comparison
    ax4 = fig.add_subplot(gs[2, :])
    ax4.hist(training_scores, bins=30, alpha=0.6, label='Training Scores', color='blue', edgecolor='black')
    ax4.hist(eval_scores, bins=30, alpha=0.6, label='Evaluation Scores', color='orange', edgecolor='black')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Training vs Evaluation Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('training_summary_report.png', dpi=150, bbox_inches='tight')
    print("Summary report saved as 'training_summary_report.png'")
    plt.close()


if __name__ == "__main__":
    # Check if Q-table exists
    if os.path.exists('q_table.npy'):
        # Load trained agent
        agent = QLearningAgent()
        agent.load_q_table('q_table.npy')
        
        # Evaluate
        env = SnakeGameEnvironment()
        state = env.reset()
        while True:
            action = agent.choose_action(state, training=False)
            state, _, done = env.step(action)
            if done:
                break
        
        print(f"Loaded trained agent with {len(agent.q_table)} states in Q-table")
        
        # Generate visualizations
        print("Generating visualizations...")
        plot_q_table_statistics(agent)
        visualize_gameplay(agent, num_games=3)
        
    else:
        print("Q-table not found. Please train the agent first using rl_agent.py")
