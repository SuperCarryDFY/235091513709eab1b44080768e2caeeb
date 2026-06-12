import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt



def read_acc(log_path):
    train, val = [], []
    train_pat = re.compile(r"step: (\d+).*?/train/accuracy: ([0-9.]+)")
    val_pat = re.compile(r"step: (\d+).*?/val/accuracy: ([0-9.]+)")

    for line in Path(log_path).read_text().splitlines():
        m = train_pat.search(line)
        if m:
            train.append((int(m.group(1)), float(m.group(2))))
            continue

        m = val_pat.search(line)
        if m:
            val.append((int(m.group(1)), float(m.group(2))))

    return train, val


def plot_curve(points, label):
    if not points:
        return
    steps, accs = zip(*points)
    plt.plot(steps, accs, marker="o", markersize=3, linewidth=1.5, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aa-log")
    parser.add_argument("--structure-log")
    parser.add_argument("--out", default="output/aa_vs_structure_acc.png")
    args = parser.parse_args()

    aa_train, aa_val = read_acc(args.aa_log)
    st_train, st_val = read_acc(args.structure_log)

    plt.figure(figsize=(9, 5))
    plot_curve(aa_train, "AA train")
    plot_curve(aa_val, "AA val")
    plot_curve(st_train, "Structure train")
    plot_curve(st_val, "Structure val")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
