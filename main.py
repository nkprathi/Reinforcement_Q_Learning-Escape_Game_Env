from env import EscapeGameGridEnv
from gameQlearning import train_q_learning_gge, visualize_q_table_gge, save_plots, save_config_info, save_special_tiles_img
import os
from datetime import datetime

special_tiles = { 
    "S": [(0, 0)],
    "G": [(10, 10)],
    "C": [(1, 3), (3, 10), (11, 11), (8, 8), (6, 11), (4, 4), (11, 7), (8,4)],
    "CF": [(1, 4), (9, 3), (3, 8), (7, 9)],
    "F": [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 9), (5, 10), (5, 11), (5, 8)],
    "B": [(2, 2), (2, 11), (7, 6), (8, 2), (11, 1)],
    "LB": [(0, 8), (0, 11), (5, 5), (5, 7), (11, 4), (9, 5)],
    "P": [(10, 2), (1, 10)], 
    "W": [(1, 2), (1, 6), (2, 4), (3, 1), (3, 6), (4, 3), (6, 2), (7, 4), (7, 5), (8, 5), (8, 1), (9, 1)] 
} 


train = True
visualize_results = True
random_initialization = True

no_episodes = 3000
learning_rate = 0.1
gamma = 0.93
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

env = EscapeGameGridEnv(grid_size=12, special_tiles=special_tiles, random_initialization=random_initialization)


# SETUP FOLDERS
task_name = input("Enter task name (press Enter for timestamp): ").strip()
if not task_name:
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    folder_name = task_name

# Consolidated Result Directory
result_dir = os.path.join("visualization", "results", folder_name)
heatmap_path = os.path.join(result_dir, "heatmap.png")

if train:
    r_hist, s_hist, e_hist, c_hist, winning_actions = train_q_learning_gge(env=env,
                         no_episodes=no_episodes,
                         epsilon=epsilon,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         alpha=learning_rate, 
                         gamma=gamma,
                         q_table_save_path="q_table_gge.npy")
    
    save_plots(r_hist, s_hist, e_hist, c_hist, winning_actions, result_dir)

    # Save Configuration Info
    hyperparams = {
        "no_episodes": no_episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "random_initialization": random_initialization
    }
    save_config_info(special_tiles, hyperparams, result_dir)
    save_special_tiles_img(special_tiles, result_dir)

if visualize_results:
    env.reset()
    visualize_q_table_gge(q_values_path="q_table_gge.npy", env=env, save_path=heatmap_path)