"""
Backtest the trained SAC agent and plot price vs SoC over the first 48 hours.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC

from envs.charging_env import EVChargingEnv

# ── load model and environment ────────────────────────────────────────────────

MODEL_PATH = "models/best_model"          # saved by EvalCallback
if not Path(f"{MODEL_PATH}.zip").exists():
    MODEL_PATH = "models/sac_smart_charger_final"

env   = EVChargingEnv(ev_id=0, input_dir="./inputs")
model = SAC.load(MODEL_PATH)

print(f"Running backtest with {MODEL_PATH} ...")

# ── run episode ───────────────────────────────────────────────────────────────

obs, _ = env.reset()
done   = False
history = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    row = env.full_data.iloc[env.current_step - 1]
    history.append(
        {
            "date":       row["date"],
            "price":      row["price"],
            "soc":        env.soc,
            "charge_rate": float(np.clip(action, 0.0, 1.0)),
            "pluggedin":  row["pluggedin"],
            "reward":     reward,
        }
    )

df = pd.DataFrame(history)
df.to_csv("test_results.csv", index=False)

print("--- Backtest finished ---")
print(f"Total reward : {df['reward'].sum():.2f} €")
print(f"Results saved: {Path('test_results.csv').resolve()}")
print(df.head(10))

# ── plot ──────────────────────────────────────────────────────────────────────

plot_df = df.head(48)
fig, ax1 = plt.subplots(figsize=(14, 6))

# Price
ax1.set_xlabel("Time")
ax1.set_ylabel("Spot price (€/kWh)", color="tab:blue")
ax1.plot(plot_df["date"], plot_df["price"], color="tab:blue", alpha=0.5, label="Price")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Charging rate shading
for i in range(len(plot_df) - 1):
    rate = plot_df["charge_rate"].iloc[i]
    if rate > 0.05:
        ax1.axvspan(
            plot_df["date"].iloc[i],
            plot_df["date"].iloc[i + 1],
            color="green",
            alpha=0.15 + 0.35 * rate,   # darker = higher rate
        )

# SoC
ax2 = ax1.twinx()
ax2.set_ylabel("Battery SoC", color="tab:red")
ax2.plot(plot_df["date"], plot_df["soc"], color="tab:red", linewidth=2, label="SoC")
ax2.set_ylim(0, 1.05)
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("SAC Charging Strategy: Price vs Battery SoC  (green = charging, darker = higher rate)")
fig.tight_layout()
plt.savefig("test_results.png", dpi=150)
plt.show()
