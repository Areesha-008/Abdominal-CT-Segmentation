import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("Exp 1 baseline", "runs/log1", "fig2_exp1.png"),
    ("Exp 2 CLAHE",    "runs/log2", "fig6_exp2.png"),
    ("Exp 3 unsharp",  "runs/log3", "fig9_exp3.png"),
]

def find_file(base):
    # log1, log1.json, ya log1.txt, jo bhi maujood ho
    for ext in ("", ".json", ".txt"):
        if os.path.isfile(base + ext):
            return base + ext
    return None

for label, base, out in RUNS:
    path = find_file(base)
    if path is None:
        print("skip (file nahi mili):", base)
        continue

    log = json.load(open(path))
    ep = [r["epoch"] for r in log]
    loss = [r["train_loss"] for r in log]
    dice = [r["val_mean_dice"] for r in log]
    b = max(range(len(dice)), key=lambda i: dice[i])

    fig, (axl, axd) = plt.subplots(1, 2, figsize=(11, 4))
    axl.plot(ep, loss, marker="o", ms=3)
    axl.set_title(f"{label}: training loss")
    axl.set_xlabel("Epoch"); axl.set_ylabel("Training loss"); axl.grid(alpha=.3)

    axd.plot(ep, dice, marker="o", ms=3)
    axd.scatter([ep[b]], [dice[b]], s=80, marker="*", color="red", zorder=5,
                label=f"best: epoch {ep[b]} ({dice[b]:.4f})")
    axd.set_title(f"{label}: validation mean Dice")
    axd.set_xlabel("Epoch"); axd.set_ylabel("Validation mean Dice")
    axd.grid(alpha=.3); axd.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out, "| source:", path)