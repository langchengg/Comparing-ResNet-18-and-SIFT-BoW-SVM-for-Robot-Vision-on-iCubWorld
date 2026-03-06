import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CNN RESULTS ----------------
cnn_df = pd.read_csv("/Users/delaynomore/Downloads/cw2/gemini_v2/cnn_results_icub.csv")

# Optional: inspect structure once
print(cnn_df.head())

# Bar plot: test accuracy per run_id, separate figure per domain
for domain, sub in cnn_df.groupby("domain"):
    plt.figure()
    plt.bar(sub["run_id"].astype(str), sub["test_acc"])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Run ID")
    plt.ylabel("Test accuracy")
    plt.title(f"CNN ResNet-18 test accuracy per run ({domain} domain)")
    plt.tight_layout()
    plt.savefig(f"figures_cnn/{domain}_cnn_test_acc_per_run.png", dpi=300)
    plt.close()

# Line plot: effect of learning rate and weight decay (example: human domain)
human = cnn_df[cnn_df["domain"] == "human"]
for wd, sub in human.groupby("wd"):
    plt.figure()
    for bs, sub_bs in sub.groupby("batch"):
        x = sub_bs["lr"]
        y = sub_bs["test_acc"]
        plt.plot(x, y, marker="o", label=f"batch={bs}")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Test accuracy")
    plt.title(f"CNN (human) – effect of learning rate (weight_decay={wd})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures_cnn/human_lr_vs_testacc_wd{wd}.png", dpi=300)
    plt.close()
