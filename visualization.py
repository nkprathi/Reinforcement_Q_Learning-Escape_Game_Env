import matplotlib.pyplot as plt
import os
import numpy as np

def calculate_sma(data, window_size):
    if len(data) >= window_size:
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return data

def save_plots(r_hist, s_hist, e_hist, c_hist, winning_actions, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Consolidated Training Plots (2x2 Grid)
    from collections import Counter
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Reward (Moving Average)
    window_size = 100  # Adjusted window size
    reward_data = calculate_sma(r_hist, window_size)
    
    axes[0, 0].plot(reward_data, color='blue')
    if len(r_hist) >= window_size:
        axes[0, 0].set_title(f"Reward Moving Average (Window={window_size})")
    else:
        axes[0, 0].set_title("Total Reward per Episode")
        
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    
    # 2. Steps (Moving Average)
    steps_data = calculate_sma(s_hist, window_size)
    
    axes[0, 1].plot(steps_data, color='orange')
    if len(s_hist) >= window_size:
        axes[0, 1].set_title(f"Steps to Goal Moving Average (Window={window_size})")
    else:
        axes[0, 1].set_title("Steps to Goal per Episode")
        
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    
    # 3. Coins (Moved to Bottom Left)
    axes[1, 0].plot(c_hist, color='purple')
    axes[1, 0].set_title("Coins Collected per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Coins")
   
    
    # 4. Winning Actions (Moved to Bottom Right)
    # 0: Up, 1: Down, 2: Right, 3: Left, None: Failed
    action_counts = Counter(winning_actions)
    actions = ["Up", "Down", "Right", "Left", "None"]
    lookup_keys = [0, 1, 2, 3, None]
    act_vals = [action_counts.get(k, 0) for k in lookup_keys]
    
    bars = axes[1, 1].bar(actions, act_vals, color='lightgreen')
    axes[1, 1].set_title("Winning Action Distribution")
    axes[1, 1].set_ylabel("Count")
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, height,
                        f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_plots.png"))
    plt.close()
    
    print(f"Consolidated graphs saved to {os.path.join(save_dir, 'training_plots.png')}")

def save_config_info(special_tiles, hyperparameters, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. Hyperparameters (Text Image)
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    param_str = "Hyperparameters:\n" + "-"*20 + "\n"
    for key, value in hyperparameters.items():
        param_str += f"{key}: {value}\n\n"
        
    plt.text(0.1, 0.9, param_str, fontsize=12, family='monospace', va='top')
    plt.title("Training Hyperparameters")
    plt.savefig(os.path.join(save_dir, "hyperparameters.png"))
    plt.close()
    print(f"Configuration info saved to {save_dir}")

def save_special_tiles_img(special_tiles, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Text content
    text_str = "Special Tiles Configuration:\n"
    text_str += "="*30 + "\n\n"
    
    for tile_type, coords in special_tiles.items():
        # Wrap coordinates if too long
        coords_str = str(coords)
        # Simple wrapping logic for display
        if len(coords_str) > 60:
            # excessive simple wrapping
            split_idx = len(coords_str) // 2
            coords_str = coords_str[:split_idx] + "\n" + " "*(len(tile_type)+2) + coords_str[split_idx:]
            
        text_str += f"{tile_type}: {coords_str}\n\n"
    
    plt.text(0.05, 0.95, text_str, fontsize=10, family='monospace', va='top')
    plt.title("Map Layout (Special Tiles)")
    
    save_path = os.path.join(save_dir, "special_tiles.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Special tiles image saved to {save_path}")
