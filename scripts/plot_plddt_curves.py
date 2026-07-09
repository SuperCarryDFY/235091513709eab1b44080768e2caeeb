import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt



def read_plddt(exp_dir, step_interval=5000):
    points = []
    ckpt_dir = Path(exp_dir) / "IntervalCheckpoints"

    for json_path in sorted(ckpt_dir.glob("step=*/sample_SwissProt-hard-500/seq_esmfold_results.json")):
        m = re.search(r"step=(\d+)", str(json_path))
        if not m:
            continue

        step = int(m.group(1))
        if step % step_interval != 0:
            continue

        data = json.loads(json_path.read_text())
        plddt = data.get("ESMFold pLDDT")
        if plddt is None:
            values = [v["plddt"] for k, v in data.items() if k.endswith("_metrics") and "plddt" in v]
            if not values:
                continue
            plddt = sum(values) / len(values)
        points.append((step, float(plddt)))

    return sorted(points)


def plot_curve(points, label, marker):
    if not points:
        return
    steps, plddts = zip(*points)
    plt.plot(steps, plddts, marker=marker, markersize=4, linewidth=1.5, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aa-dir", default="output/AATokenPrediction")
    parser.add_argument("--structure-dir", default="output/StructureTokenPrediction")
    parser.add_argument("--step-interval", type=int, default=5000)
    parser.add_argument("--out", default="output/aa_vs_structure_plddt.png")
    args = parser.parse_args()

    aa_plddt = read_plddt(args.aa_dir, args.step_interval)
    st_plddt = read_plddt(args.structure_dir, args.step_interval)

    plt.figure(figsize=(9, 5))
    plot_curve(st_plddt, "Structure (pLDDT)", "o")
    plot_curve(aa_plddt, "Sequence (pLDDT)", "s")
    plt.xlabel("Training Steps")
    plt.ylabel("ESMFold pLDDT")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
