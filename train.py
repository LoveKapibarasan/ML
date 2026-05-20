"""
Train an EV smart-charging agent with SAC (Soft Actor-Critic).

SAC is an off-policy, maximum-entropy algorithm well-suited for continuous
control.  Compared with PPO it offers:
  • Higher sample efficiency via experience replay
  • Automatic entropy tuning (prevents premature convergence)
  • Naturally handles continuous charging-rate actions
"""

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.charging_env import EVChargingEnv


def train():
    env = Monitor(EVChargingEnv(ev_id=0, input_dir="./inputs"))
    eval_env = Monitor(EVChargingEnv(ev_id=0, input_dir="./inputs"))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        # --- learning dynamics ---
        learning_rate=3e-4,
        gamma=0.99,  # discount factor
        tau=0.005,  # soft-update coefficient for target nets
        # --- replay buffer ---
        buffer_size=200_000,
        learning_starts=2_000,  # steps before first gradient update
        batch_size=256,
        # --- update schedule ---
        train_freq=1,
        gradient_steps=1,
        # --- entropy ---
        ent_coef="auto",  # automatic entropy-coefficient tuning
        target_entropy="auto",
        # --- network ---
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./sac_ev_logs/",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./sac_ev_logs/",
        eval_freq=5_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )

    print("Training started (SAC)...")
    model.learn(total_timesteps=500_000, callback=eval_cb, progress_bar=True)

    model.save("models/sac_smart_charger_final")
    print("Model saved → models/sac_smart_charger_final")


if __name__ == "__main__":
    train()
