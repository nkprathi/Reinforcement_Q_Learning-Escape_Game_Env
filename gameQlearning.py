import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from visualization import save_plots, save_config_info, save_special_tiles_img

def train_q_learning_gge(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, q_table_save_path="q_table_gge.npy"):
    
    q_table = np.zeros((env.grid_size, env.grid_size, 256, env.action_space.n))
    
    reward_history = []
    steps_history = []
    epsilon_history = []
    coins_history = []
    winning_actions = []


    for episode in range(no_episodes):
        state_full, _ = env.reset()
        state = state_full[:2] 
        coin_mask = state_full[2]
        
        total_reward = 0

        
        steps = 0
        done = False
        
        last_action = None

        while not done:
            if env.random_initialization:
                # Epsilon greedy logic same as before...
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    x, y = state
                    q_values = q_table[x, y, coin_mask, :]
                    action = np.argmax(q_values)
            else:
                x, y = state
                q_values = q_table[x, y, coin_mask, :]
                action = np.argmax(q_values)
                
            next_state_full, reward, done, _ = env.step(action)
            env.render() 
            nx, ny = next_state_full[:2]
            n_coin_mask = next_state_full[2]
            
            # Q-Update
            x, y = state
            q_table[x, y, coin_mask, action] += alpha * (reward + gamma * np.max(q_table[nx, ny, n_coin_mask]) - q_table[x, y, coin_mask, action])
            
            state = [nx, ny]
            coin_mask = n_coin_mask
            total_reward += reward
            steps += 1
            last_action = action
            
        if reward == 300:
            winning_actions.append(last_action)
        else:
            winning_actions.append(None)
            
        reward_history.append(total_reward)
        steps_history.append(steps)
        epsilon_history.append(epsilon)
        coins_history.append(sum(env.coins))
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print("Training finished.")
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table for GameGrid.")
    return reward_history, steps_history, epsilon_history, coins_history, winning_actions

def visualize_q_table_gge(q_values_path, env, save_path=None, actions=["Up", "Down", "Right", "Left"]):

    try:
        q_table = np.load(q_values_path)
        # build mask from env grid (True = hide). Masks 'W' (walls) and 'R' (river)
        grid_labels = env.grid.astype(str)
        mask_cells = np.isin(grid_labels, ['W', 'F', 'G', 'B', 'LB'])

        _, axes = plt.subplots(1, 4, figsize=(25, 5))
        for i, action in enumerate(actions):
            ax = axes[i]
            # Visualize the MAX Q-value across all coin states for this action
            # This shows the "best case" potential of each cell (e.g. if you had the coins)
            #heatmap_data = np.max(q_table[:, :, :, i], axis=2).astype(float)
            # Treat 0.0 (unvisited) as -inf so that real negative values (e.g. -10) are chosen by max()
            masked_q = np.where(np.isclose(q_table[:, :, :, i], 0.0), -np.inf, q_table[:, :, :, i])
            heatmap_data = np.max(masked_q, axis=2)
            # Replace -inf back with 0 (truly unvisited states)
            heatmap_data[heatmap_data == -np.inf] = 0
            sns.heatmap(heatmap_data, mask=mask_cells, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar=False, annot_kws={"size": 8}, linewidths=0.5, linecolor="black", vmin=-20, vmax=305)
            ax.set_title(f'Action: {action}')
            
            # annotate masked cells 
            rows, cols = heatmap_data.shape
            
            for r in range(rows):
                for c in range(cols):
                    label = grid_labels[r, c]
                    
                    # 1. Handle Masked Cells (Walls, Bombs, etc.) -> Black Text
                    if mask_cells[r, c]:
                        ax.text(c + 0.5, r + 0.5, label, ha='center', va='center', color='black', fontsize=10, fontweight='bold')
                    
                    # 2. Handle Unmasked Special Tiles (Keys, Doors) -> White Text
                    elif label in ['C', 'CF', 'S', 'G', 'B', 'F', 'P']:
                        ax.text(c + 0.1, r + 0.1, label, ha='left', va='top', color='magenta', fontsize=7, fontweight='semibold')

        plt.tight_layout()
        
        final_save_path = save_path if save_path else "/Users/prathi/Desktop/heatmap.png"
        directory = os.path.dirname(final_save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(final_save_path)
        print(f"Heatmap saved to {final_save_path}")
        plt.show()
    except FileNotFoundError:
        print("Saved Q-table not found for GameGrid! Train first or check file path.")