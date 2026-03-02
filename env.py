import gymnasium as gym
import numpy as np
import pygame
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class EscapeGameGridEnv(gym.Env):
    def __init__(self, grid_size=12, special_tiles=None, random_initialization=False):
        super(EscapeGameGridEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 60
        self.state = None
        self.reward = 0
        self.info = {}
        self.done = False
        self.random_initialization = random_initialization

        # Initialize empty grid
        self.grid = np.full((grid_size, grid_size), "E", dtype=object)
        self.special_tiles = special_tiles
        # Fill grid based on special_tiles dictionary
        if special_tiles is not None:
            for tile_type, coords in special_tiles.items():
                for (x, y) in coords:
                    self.grid[x, y] = tile_type
        
        # Identify locations of key tiles for logic
        self.locs = {}
        for label in ['S', 'G', 'B', 'C', 'CF', 'LB', 'F']:
            arr = np.argwhere(self.grid == label)
            if len(arr) > 0:
                self.locs[label] = tuple(arr[0])

        # Store explicit coin coordinates for indexing
        self.coin_coords = []
        if special_tiles and "C" in special_tiles:
            self.coin_coords = special_tiles["C"] # List of tuples
        
        self.action_space = gym.spaces.Discrete(4)
        #self.action_space = gym.spaces.Discrete(4)
        # Observation: [x, y, coin_mask]
        # x, y < grid_size; coin_mask < 2^NumCoins (256 for 8 coins)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3,), dtype=np.int32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))

        # Coins collected by agent
        self.num_coins = len(self.coin_coords) if self.coin_coords else 0
        self.coins = [0] * self.num_coins

        self.images = {
            "B": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "Bomb.png")),
            "S": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "Start.png")),
            "G": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "Goal.png")),
            "AGENT": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "Human_icon.png")),
            "C": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "GoldCoin.png")),
            "F": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "ElectricFence.png")),
            "CF": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "CrumbledFloor.png")),
            "W": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "Wall.png")),
            "P": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "Power.png")),
            "LB": pygame.image.load(os.path.join(os.path.dirname(__file__), "visualization", "images", "LaserBeam.png")),
        }

    def reset(self):
        
        if self.random_initialization:
            empties = np.argwhere(self.grid == "E")
            start = empties[np.random.randint(len(empties))]
            self.state = [start[0], start[1]]
        else:
            self.state = [self.locs['S'][0], self.locs['S'][1]]

        if hasattr(self, 'special_tiles') and self.special_tiles is not None:
            for tile_type, coords in self.special_tiles.items(): #category loop
                for (x, y) in coords: #placement loop
                    self.grid[x, y] = tile_type

        self.coins = [0] * self.num_coins
        self.done = False
        self.reward = 0
        

        coin_mask = int("".join(map(str, self.coins)), 2)
        
        self.info = {
            "Distance to goal": np.linalg.norm(np.array(self.state)-np.array(self.locs['G'])),
            "Coins": self.coins.copy()
        }
        return np.array(self.state + [coin_mask]), self.info

    def step(self, action):
        coin_mask = int("".join(map(str, self.coins)), 2)
        if self.done:
            return np.array(self.state + [coin_mask]), self.reward, self.done, self.info

        x, y = self.state
        # Compute next position
        if action == 0 and x > 0: #Movement Engine
            nx = x - 1; ny = y
        elif action == 1 and x < self.grid_size-1:
            nx = x + 1; ny = y
        elif action == 2 and y < self.grid_size-1:
            nx = x; ny = y + 1
        elif action == 3 and y > 0:
            nx = x; ny = y - 1
        else:
            nx, ny = x, y  # no move if out of bounds

        target = self.grid[nx, ny]
        self.reward = -1 

        # Interaction logic
        if target == "W":     
            nx, ny = x, y
            self.reward = -10 
        elif target == "F":     
            nx, ny = x, y       
            self.reward = -10
        elif target == "LB":    
            self.reward = -10
        elif target == "B":     
            self.reward = -50
            self.done = True
        elif target == "G":     
            if sum(self.coins) >= 4:
                self.reward = 300
                self.done = True

        elif target == "C":
            coin_idx = self.coin_coords.index((nx, ny))
            self.coins[coin_idx] = 1
            self.grid[nx, ny] = "E"
            self.reward = 10

        elif target == "P":     
            self.grid[nx, ny] = "E"  
            self.reward = 20
        elif target == "CF":     
            self.grid[nx, ny] = "W"  
            self.reward = -5
        elif target == "S":     
            pass
        
        self.state = [nx, ny]
        coin_mask = int("".join(map(str, self.coins)), 2)
        
        self.info["Distance to goal"] = np.linalg.norm(np.array(self.state)-np.array(self.locs['G']))
        self.info["Coins"] = self.coins.copy()

        obs = np.array(self.state + [coin_mask])
        return obs, self.reward, self.done, self.info


    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.screen.fill((255, 255, 255))

        colors = dict.fromkeys(["E", "S", "G", "B", "C", "CF", "W", "F", "LB", "P"], (255, 255, 255))

        # draw cells
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                tile_type = str(self.grid[row, col])  
                color = colors.get(tile_type, (230, 230, 230))
                cell_rect = pygame.Rect(col * self.cell_size,
                                        row * self.cell_size,
                                        self.cell_size,
                                        self.cell_size)
                # fill cell
                pygame.draw.rect(self.screen, color, cell_rect)

                # draw image if available (full cell)
                if hasattr(self, "images") and tile_type in self.images:
                    img = pygame.transform.scale(self.images[tile_type], (self.cell_size, self.cell_size))
                    self.screen.blit(img, cell_rect.topleft)

        # draw vertical grid lines
        width_px = self.cell_size * self.grid_size
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, width_px), 1)

        # draw horizontal grid lines
        for j in range(self.grid_size + 1):
            y = j * self.cell_size
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (width_px, y), 1)

        # draw agent outline on top
        if self.state is not None:
            agent_rect = pygame.Rect(self.state[1] * self.cell_size,
                                     self.state[0] * self.cell_size,
                                     self.cell_size,
                                     self.cell_size)
            
            if "AGENT" in self.images:
                # 1. Get the raw image using the key we defined earlier (e.g., "AGENT")
                agent_img_raw = self.images["AGENT"]
                
                # 2. Scale the image to fit the cell size
                img = pygame.transform.scale(agent_img_raw, (self.cell_size, self.cell_size))
                
                # 3. Blit (draw) the image at the top-left corner of the agent's cell
                self.screen.blit(img, agent_rect.topleft)

            pygame.draw.rect(self.screen, (0, 0, 0), agent_rect, 3)

        pygame.display.flip()


    def close(self):
        pygame.quit()
