# =========================
# file: plot_metrics.py  (offline visualization)
# =========================
import argparse
import csv
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="runs/dqn_metrics.csv", help="CSV emitted by MetricsLogger")
    ap.add_argument("--save", type=str, default="", help="Optional PNG path")
    args = ap.parse_args()

    ep, dur, rew, loss = [], [], [], []
    with open(args.csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            ep.append(int(row["episode"]))
            dur.append(float(row["duration"]))
            rew.append(float(row["reward"]))
            val = row.get("loss", "")
            try:
                loss.append(float(val))
            except Exception:
                loss.append(float("nan"))

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(ep, dur)
    plt.title("Episode Duration")
    plt.xlabel("Episode"); plt.ylabel("Ticks")

    plt.subplot(2,1,2)
    plt.plot(ep, rew)
    plt.title("Episode Reward")
    plt.xlabel("Episode"); plt.ylabel("Reward")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"saved to {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()