import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.charging_env import EVChargingEnv
import matplotlib.pyplot as plt

env = EVChargingEnv(ev_id=0, input_dir="./inputs")

model_path = "models/ppo_smart_charger_v1"
model = PPO.load(model_path)
obs, _ = env.reset()
done = False
history = []

print(f"Running Backtest with {model_path}...")

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    current_row = env.full_data.iloc[env.current_step - 1]
    current_row = env.full_data.iloc[env.current_step - 1]
    history.append({
        "date": current_row["date"],
        "price": current_row["price"],
        "soc": env.soc,
        "action": action,
        "pluggedin": current_row["pluggedin"]
    })

df_res = pd.DataFrame(history)
df_res.to_csv("test_results.csv", index=False)
print("--- Test finished! ---")
print(f"Results saved to: {pd.io.common.os.path.abspath('test_results.csv')}")

print("\nSample Output (First 10 rows):")
print(df_res.head(10))

df = pd.read_csv("./test_results.csv", parse_dates=['date'])


plot_df = df.head(48) 

fig, ax1 = plt.subplots(figsize=(12, 6))

color_price = 'tab:blue'
ax1.set_xlabel('Time')
ax1.set_ylabel('Price (€/kWh)', color=color_price)
ax1.plot(plot_df['date'], plot_df['price'], color=color_price, label='Market Price', alpha=0.5)
ax1.tick_params(axis='y', labelcolor=color_price)

for i in range(len(plot_df)-1):
    if plot_df['action'].iloc[i] == 1:
        ax1.axvspan(plot_df['date'].iloc[i], plot_df['date'].iloc[i+1], color='green', alpha=0.2)

ax2 = ax1.twinx() 
color_soc = 'tab:red'
ax2.set_ylabel('SoC (Battery %)', color=color_soc)
ax2.plot(plot_df['date'], plot_df['soc'], color=color_soc, linewidth=2, label='EV SoC')
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis='y', labelcolor=color_soc)

plt.title('AI Charging Strategy: Price vs Battery Level')
fig.tight_layout()
plt.show()