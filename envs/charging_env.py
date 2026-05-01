import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path

class EVChargingEnv(gym.Env):
    def __init__(self, ev_id, input_dir="./inputs"): 
        super(EVChargingEnv, self).__init__()
        
        self.full_data, self.demand_data = self._load_and_merge_data(ev_id, input_dir)
        
        self.battery_capacity = 50.0 
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        
        self.reset()

    def _load_and_merge_data(self, ev_id, input_dir):
        path = Path(input_dir).resolve()
        df = pd.read_csv(path / f"processed_data_{ev_id}.csv", parse_dates=['date'])
        df_demand = pd.read_csv(path / f"processed_demand_{ev_id}.csv", 
                                parse_dates=['arrival_time', 'departure_time'])
        return df, df_demand

    def _get_obs(self):
        row = self.full_data.iloc[self.current_step]
        demand = self.demand_data.iloc[self.current_demand_idx]
        
        time_left = (demand['departure_time'] - row['date']).total_seconds() / 3600
        
        return np.array([
            min(row['price'] / 0.1, 1.0),           
            (row['temp_c'] + 10) / 50.0,            
            min(row['radiation_wm2'] / 1000.0, 1.0), 
            min(row['load'] / 20.0, 1.0),            
            min(row['pv'] / 10.0, 1.0),              
            float(row['is_holiday_x']),              
            float(row['pluggedin']),                 
            float(self.soc),                        
            min(max(time_left / 24.0, 0), 1),       
            float(demand['target_soc']),         
            min(row['powerrating'] / 22.0, 1.0),     
            min(row['price_3h_future'] / 0.1, 1.0)   
        ], dtype=np.float32)

    def _calculate_reward(self, action, row, is_departure, demand):
        reward = 0.0
        if action == 1 and row['pluggedin']:
            net_power = max(0, row['powerrating'] - row['pv'])
            cost = (net_power * row['price']) / 1.0 
            reward -= cost
            
        if is_departure:
            if self.soc >= (demand['target_soc'] - 0.02):
                reward += 20.0
            else:
                reward -= 100.0 * (demand['target_soc'] - self.soc)
        return reward

    def step(self, action):
        row = self.full_data.iloc[self.current_step]
        demand = self.demand_data.iloc[self.current_demand_idx]
        
        is_departure = (row['date'] >= demand['departure_time'])
        reward = self._calculate_reward(action, row, is_departure, demand)
        
        if action == 1 and row['pluggedin']:
            added = (row['powerrating'] * 1.0) / self.battery_capacity
            self.soc = min(1.0, self.soc + added)
            
        if is_departure:
            if self.current_demand_idx < len(self.demand_data) - 1:
                self.current_demand_idx += 1
                self.soc = self.demand_data.iloc[self.current_demand_idx]['initial_soc']

        self.current_step += 1
        done = self.current_step >= len(self.full_data) - 1
        
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_demand_idx = 0
        self.soc = self.demand_data.iloc[0]['initial_soc']
        return self._get_obs(), {}