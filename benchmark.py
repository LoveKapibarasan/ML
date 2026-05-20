"""
Benchmark: SAC agent vs baseline strategies
Generates benchmark_results/ with multiple plots and a CSV summary.

Strategies compared
-------------------
  sac          : trained SAC agent (deterministic)
  always_on    : charge at full power whenever plugged in
  threshold    : charge only when price < median price
  greedy_cheap : sort hours by price and charge the cheapest ones first
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from envs.charging_env import EVChargingEnv

OUT_DIR = Path("benchmark_results")
OUT_DIR.mkdir(exist_ok=True)

BATTERY_KWH = 50.0
CHARGE_THRESHOLD = 0.05
STYLE = {
    "sac": dict(color="#2196F3", label="SAC (ours)", zorder=4),
    "always_on": dict(color="#F44336", label="Always-on baseline", zorder=2),
    "threshold": dict(color="#FF9800", label="Price-threshold", zorder=3),
    "greedy_cheap": dict(color="#9C27B0", label="Greedy cheapest-hours", zorder=3),
}

# ── helpers ───────────────────────────────────────────────────────────────────


def _record(env, action, reward, pre_soc, pre_demand) -> dict:
    """Build one step record. Must be called AFTER env.step()."""
    row = env.full_data.iloc[env.current_step - 1]
    rate = float(np.clip(np.asarray(action).flat[0], 0.0, 1.0))
    pluggedin = str(row["pluggedin"]).lower() not in ("false", "0", "")

    # Reconstruct SoC at departure (before session reset in env.step)
    if rate > CHARGE_THRESHOLD and pluggedin:
        dep_soc = min(1.0, pre_soc + rate * float(row["powerrating"]) / BATTERY_KWH)
    else:
        dep_soc = pre_soc

    is_dep = row["date"] >= pre_demand["departure_time"]
    return {
        "date": row["date"],
        "price": row["price"],
        "soc": env.soc,  # current SoC (may be new session's initial)
        "dep_soc": dep_soc,  # SoC achieved at departure (before reset)
        "rate": rate,
        "powerrating": row["powerrating"],
        "reward": reward,
        "pluggedin": pluggedin,
        "target_soc": float(pre_demand["target_soc"]),  # session that just ended/ran
        "departure": pre_demand["departure_time"],
        "is_departure": is_dep,
    }


def _run_sac(env: EVChargingEnv, model: SAC) -> list[dict]:
    obs, _ = env.reset()
    done, records = False, []
    while not done:
        pre_soc = env.soc
        pre_demand = env.demand_data.iloc[env.current_demand_idx]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        records.append(_record(env, action, reward, pre_soc, pre_demand))
    return records


def _run_always_on(env: EVChargingEnv) -> list[dict]:
    obs, _ = env.reset()
    done, records = False, []
    while not done:
        pre_soc = env.soc
        pre_demand = env.demand_data.iloc[env.current_demand_idx]
        action = np.array([1.0])
        obs, reward, done, _, _ = env.step(action)
        records.append(_record(env, action, reward, pre_soc, pre_demand))
    return records


def _run_threshold(env: EVChargingEnv, pct: float = 0.5) -> list[dict]:
    """Charge at full power when price < pct-quantile, else off."""
    threshold = float(env.full_data["price"].quantile(pct))
    obs, _ = env.reset()
    done, records = False, []
    while not done:
        pre_soc = env.soc
        pre_demand = env.demand_data.iloc[env.current_demand_idx]
        cur_price = float(env.full_data.iloc[env.current_step]["price"])
        rate = 1.0 if cur_price <= threshold else 0.0
        action = np.array([rate])
        obs, reward, done, _, _ = env.step(action)
        records.append(_record(env, action, reward, pre_soc, pre_demand))
    return records


def _run_greedy(env: EVChargingEnv) -> list[dict]:
    """Pre-select the cheapest hours within each session to meet target SoC."""
    full = env.full_data.copy().reset_index(drop=True)
    demand = env.demand_data.copy()

    charge_hours: set[int] = set()
    for _, sess in demand.iterrows():
        mask = (full["date"] >= sess["arrival_time"]) & (
            full["date"] < sess["departure_time"]
        )
        hours = full[mask].copy()
        if hours.empty:
            continue
        kw_needed = (
            max(0.0, float(sess["target_soc"]) - float(sess["initial_soc"]))
            * BATTERY_KWH
        )
        accumulated = 0.0
        for idx, h in hours.sort_values("price").iterrows():
            pw = float(h["powerrating"]) if float(h["powerrating"]) > 0 else 11.0
            accumulated += pw
            charge_hours.add(int(idx))
            if accumulated >= kw_needed:
                break

    obs, _ = env.reset()
    done, records = False, []
    step = 0
    while not done:
        pre_soc = env.soc
        pre_demand = env.demand_data.iloc[env.current_demand_idx]
        rate = 1.0 if step in charge_hours else 0.0
        action = np.array([rate])
        obs, reward, done, _, _ = env.step(action)
        records.append(_record(env, action, reward, pre_soc, pre_demand))
        step += 1
    return records


def _to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["departure"] = pd.to_datetime(df["departure"])
    df["cost_eur"] = df["rate"] * df["powerrating"] * df["price"]
    df["cost_eur"] = df["cost_eur"].clip(lower=0)  # negative = earned money
    return df


def _departure_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rows flagged as departure events; checks dep_soc vs target."""
    dep = df[df["is_departure"]].copy()
    dep["met"] = dep["dep_soc"] >= (dep["target_soc"] - 0.02)
    return dep


# ── plots ─────────────────────────────────────────────────────────────────────


def plot_price_soc_charging(dfs: dict[str, pd.DataFrame], days: int = 7) -> None:
    """Price timeline with SoC and charging rate for first N days."""
    names = list(dfs.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(16, 4 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    ref = dfs[names[0]]
    cutoff = ref["date"].iloc[0] + pd.Timedelta(days=days)
    xticks = pd.date_range(ref["date"].iloc[0], cutoff, freq="D")

    for ax, name in zip(axes, names, strict=False):
        df = dfs[name][dfs[name]["date"] <= cutoff]
        info = STYLE[name]

        ax2 = ax.twinx()

        # Charging rate shading
        for i in range(len(df) - 1):
            if df["rate"].iloc[i] > CHARGE_THRESHOLD:
                ax.axvspan(
                    df["date"].iloc[i],
                    df["date"].iloc[i + 1],
                    alpha=0.15 + 0.3 * df["rate"].iloc[i],
                    color=info["color"],
                    zorder=1,
                )

        ax.plot(
            df["date"],
            df["price"],
            color="#607D8B",
            alpha=0.7,
            linewidth=1,
            label="Spot price (€/kWh)",
        )
        ax.axhline(0, color="#607D8B", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel("Spot price (€/kWh)", fontsize=9)

        ax2.plot(
            df["date"],
            df["soc"],
            color=info["color"],
            linewidth=2,
            label="SoC",
            zorder=5,
        )
        ax2.set_ylabel("Battery SoC", color=info["color"], fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor=info["color"])

        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.set_xticks(xticks)
        ax.set_xticklabels([d.strftime("%m/%d") for d in xticks], fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Charging Strategy Comparison — first {days} days\n"
        "(shaded = charging, darker = higher rate)",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_price_soc_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✓ 01_price_soc_comparison.png")


def plot_cumulative_cost(dfs: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, df in dfs.items():
        info = STYLE[name]
        cum = df["cost_eur"].cumsum()
        ax.plot(
            df["date"],
            cum,
            label=info["label"],
            color=info["color"],
            linewidth=2,
            zorder=info["zorder"],
        )

    ax.set_title("Cumulative Charging Cost Over Time", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative cost (€)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_cumulative_cost.png", dpi=150)
    plt.close(fig)
    print("  ✓ 02_cumulative_cost.png")


def plot_cost_bar(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = [STYLE[n]["color"] for n in summary.index]
    labels = [STYLE[n]["label"] for n in summary.index]

    # Total cost
    ax = axes[0]
    bars = ax.bar(labels, summary["total_cost_eur"], color=colors, edgecolor="white")
    ax.set_title("Total Charging Cost (€)", fontsize=12)
    ax.set_ylabel("€")
    for bar, val in zip(bars, summary["total_cost_eur"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"€{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # SoC fulfillment rate
    ax = axes[1]
    pcts = summary["soc_met_pct"]
    bars = ax.bar(labels, pcts, color=colors, edgecolor="white")
    ax.set_title("Departure SoC Target Met (%)", fontsize=12)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, pcts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Strategy Comparison Summary", fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_cost_and_fulfillment.png", dpi=150)
    plt.close(fig)
    print("  ✓ 03_cost_and_fulfillment.png")


def plot_price_vs_rate(dfs: dict[str, pd.DataFrame]) -> None:
    """Scatter: spot price vs charging rate (plugged-in hours only)."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4), sharey=True)
    if len(dfs) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, dfs.items(), strict=False):
        info = STYLE[name]
        plugged = df[df["pluggedin"]]
        ax.scatter(
            plugged["price"],
            plugged["rate"],
            c=info["color"],
            alpha=0.3,
            s=10,
            rasterized=True,
        )
        ax.set_title(info["label"], fontsize=10)
        ax.set_xlabel("Spot price (€/kWh)", fontsize=9)
        ax.set_ylabel("Charging rate" if ax == axes[0] else "", fontsize=9)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("Spot Price vs Charging Rate (plugged-in hours)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "04_price_vs_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_price_vs_rate.png")


def plot_training_reward() -> None:
    """Plot eval reward curve from EvalCallback log."""
    log_files = sorted(Path("sac_ev_logs").rglob("evaluations.npz"))
    if not log_files:
        print("  ⚠ No evaluations.npz found — skipping training curve")
        return

    data = np.load(log_files[-1])
    steps = data["timesteps"]
    rews = data["results"].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, rews, color="#2196F3", linewidth=2)
    ax.fill_between(
        steps, rews - rews.std(), rews + rews.std(), color="#2196F3", alpha=0.15
    )
    ax.set_title("SAC Training — Evaluation Reward Curve", fontsize=13)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Reward (€)")
    ax.grid(alpha=0.3)
    ax.axhline(
        rews[-1],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Final: {rews[-1]:.1f} €",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "00_training_curve.png", dpi=150)
    plt.close(fig)
    print("  ✓ 00_training_curve.png")


def plot_hourly_price_preference(dfs: dict[str, pd.DataFrame]) -> None:
    """Average charging rate per hour-of-day: shows when each strategy charges."""
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.arange(24)

    for name, df in dfs.items():
        info = STYLE[name]
        plugged = df[df["pluggedin"]].copy()
        plugged["hour"] = plugged["date"].dt.hour
        avg = plugged.groupby("hour")["rate"].mean().reindex(hours, fill_value=0)
        ax.plot(
            hours,
            avg,
            marker="o",
            markersize=4,
            label=info["label"],
            color=info["color"],
            linewidth=2,
            zorder=info["zorder"],
        )

    ax.set_title(
        "Average Charging Rate by Hour of Day (plugged-in sessions)", fontsize=12
    )
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Mean charging rate [0–1]")
    ax.set_xticks(hours)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "05_hourly_charging_pattern.png", dpi=150)
    plt.close(fig)
    print("  ✓ 05_hourly_charging_pattern.png")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading model and environment...")
    env = EVChargingEnv(ev_id=0, input_dir="./inputs")
    model = SAC.load("models/best_model")

    print("Running strategies...")
    raw = {
        "sac": _run_sac(env, model),
        "always_on": _run_always_on(env),
        "threshold": _run_threshold(env),
        "greedy_cheap": _run_greedy(env),
    }
    dfs = {k: _to_df(v) for k, v in raw.items()}

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for name, df in dfs.items():
        dep = _departure_stats(df)
        met = dep["met"].mean() * 100 if len(dep) > 0 else 0.0
        rows.append(
            {
                "strategy": name,
                "label": STYLE[name]["label"],
                "total_cost_eur": df["cost_eur"].sum(),
                "total_reward": df["reward"].sum(),
                "soc_met_pct": met,
                "n_steps": len(df),
            }
        )

    summary = pd.DataFrame(rows).set_index("strategy")
    summary.to_csv(OUT_DIR / "summary.csv")

    # ── Savings vs always-on ──────────────────────────────────────────────────
    baseline_cost = summary.loc["always_on", "total_cost_eur"]
    print("\n─── Benchmark Summary ─────────────────────────────────────────")
    print(f"{'Strategy':<25} {'Cost (€)':>10} {'Savings (€)':>12} {'SoC met':>9}")
    print("─" * 62)
    for name, row in summary.iterrows():
        savings = baseline_cost - row["total_cost_eur"]
        print(
            f"{row['label']:<25} {row['total_cost_eur']:>10.2f}"
            f" {savings:>+12.2f}   {row['soc_met_pct']:>7.1f}%"
        )
    print("─" * 62)

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_training_reward()
    plot_price_soc_charging(dfs, days=14)
    plot_cumulative_cost(dfs)
    plot_cost_bar(summary)
    plot_price_vs_rate(dfs)
    plot_hourly_price_preference(dfs)

    print(f"\nAll outputs saved to {OUT_DIR.resolve()}/")
    print("  summary.csv + 6 PNG plots")


if __name__ == "__main__":
    main()
