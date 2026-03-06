import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CNN RESULTS ----------------
cnn_results_df = pd.read_csv("/Users/delaynomore/Downloads/cw2/gemini_v2/cnn_results_icub.csv")

# Optional: inspect structure once
print(cnn_results_df.head())

# Bar plot: test accuracy per run_id, separate figure per domain
for domain, domain_data in cnn_results_df.groupby("domain"):
    plt.figure()
    plt.bar(domain_data["run_id"].astype(str), domain_data["test_acc"])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Run ID")
    plt.ylabel("Test accuracy")
    plt.title(f"CNN ResNet-18 test accuracy per run ({domain} domain)")
    plt.tight_layout()
    plt.savefig(f"figures_cnn/{domain}_cnn_test_acc_per_run.png", dpi=300)
    plt.close()

# Line plot: effect of learning rate and weight decay (example: human domain)
human_domain_data = cnn_results_df[cnn_results_df["domain"] == "human"]
for weight_decay_value, weight_decay_subset in human_domain_data.groupby("wd"):
    plt.figure()
    for batch_size_value, batch_size_subset in weight_decay_subset.groupby("batch"):
        learning_rates = batch_size_subset["lr"]
        test_accuracies = batch_size_subset["test_acc"]
        plt.plot(learning_rates, test_accuracies, marker="o", label=f"batch={batch_size_value}")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Test accuracy")
    plt.title(f"CNN (human) – effect of learning rate (weight_decay={weight_decay_value})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures_cnn/human_lr_vs_testacc_wd{weight_decay_value}.png", dpi=300)
    plt.close()
