import gymnasium as gym
from stable_baselines3 import PPO
from envs.charging_env import EVChargingEnv

def train():
    env = EVChargingEnv(ev_id=0, input_dir="./inputs")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        device="cuda", 
        tensorboard_log="./ppo_ev_logs/"
    )

    print("Training started...")
    model.learn(total_timesteps=1000000, progress_bar=True)

    model.save("models/ppo_smart_charger_v1")
    print("Model saved.")

if __name__ == "__main__":
    train()