"""
bow_sift_svm_icub.py
Traditional CV pipeline: SIFT + BoW + SVM/KNN.

"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torchvision import datasets

# ===== Global configuration =====
DATA_ROOT = Path("/Users/delaynomore/Downloads/iCubWorld1.0")
DOMAINS = ["human", "robot"]

# Hyperparameters to explore
PARAMS_GRID = {
    "vocab_size": [50, 100],
    "classifier": ["svm", "knn"],
    "C": [0.1, 1.0],  # Only for SVM
    "n_neighbors": [5, 10]  # Only for KNN
}

RESULTS_CSV = "bow_sift_results_icub.csv"
FIG_DIR = Path("figures_bow_sift")
FIG_DIR.mkdir(exist_ok=True)


# ===== Data utilities =====
def get_image_paths_and_labels(root_dir):
    dataset = datasets.ImageFolder(root_dir)
    paths = [s[0] for s in dataset.samples]
    labels = [s[1] for s in dataset.samples]
    return np.array(paths), np.array(labels), dataset.classes


# ===== SIFT & BoW =====
def compute_sift_descriptors(image_paths, max_features=500):
    sift = cv2.SIFT_create(nfeatures=max_features)
    desc_list = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            desc_list.append(None)
            continue
        _, descriptors = sift.detectAndCompute(img, None)
        desc_list.append(descriptors)
    return desc_list


def build_vocabulary(descriptors_list, vocab_size, max_samples=100000):
    all_desc = [d for d in descriptors_list if d is not None]
    if not all_desc: raise RuntimeError("No descriptors found.")

    # Stack and Force Float64 for Scikit-Learn compatibility
    all_desc = np.vstack(all_desc).astype(np.float64)

    if len(all_desc) > max_samples:
        idx = np.random.choice(len(all_desc), max_samples, replace=False)
        all_desc = all_desc[idx]

    print(f"  Clustering {len(all_desc)} descriptors -> {vocab_size} words...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=1000, random_state=42, n_init='auto')
    kmeans.fit(all_desc)
    return kmeans


def compute_bow_features(desc_list, kmeans, vocab_size):
    features = []
    for desc in desc_list:
        if desc is None:
            features.append(np.zeros(vocab_size))
            continue

        # FIX: Explicit cast to float64 before prediction
        words = kmeans.predict(desc.astype(np.float64))

        hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
        norm = np.linalg.norm(hist)
        if norm > 0: hist = hist / norm  # L2 norm usually better for SVM
        features.append(hist)
    return np.stack(features)


# ===== Visualization =====
def save_error_cases(domain, run_id, X_test, y_test, y_pred, test_paths, class_names):

    errors = np.where(y_test != y_pred)[0]
    if len(errors) == 0: return

    num_plot = min(5, len(errors))
    fig, axes = plt.subplots(1, num_plot, figsize=(15, 3))
    if num_plot == 1: axes = [axes]

    for i, idx in enumerate(errors[:num_plot]):
        path = test_paths[idx]
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            true_lbl = class_names[y_test[idx]]
            pred_lbl = class_names[y_pred[idx]]

            axes[i].imshow(img)
            axes[i].set_title(f"T: {true_lbl}\nP: {pred_lbl}", color='red', fontsize=10)
            axes[i].axis('off')

    plt.suptitle(f"{domain} Run {run_id} Errors")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_errors.png")
    plt.close()


# ===== Experiment Logic =====
def run_experiment_logic():
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "run_id", "vocab", "clf_type", "param", "val_acc", "test_acc"])

        run_id = 0
        for domain in DOMAINS:
            print(f"\nProcessing Domain: {domain}")

            # 1. Load Data
            train_full_paths, train_full_labels, classes = get_image_paths_and_labels(DATA_ROOT / domain / "train")
            test_paths, test_labels, _ = get_image_paths_and_labels(DATA_ROOT / domain / "test")

            # 2. Split Train into Train/Val (Essential for hyperparam tuning)
            train_paths, val_paths, train_y, val_y = train_test_split(
                train_full_paths, train_full_labels, test_size=0.2, stratify=train_full_labels, random_state=42
            )

            # 3. Compute Descriptors (expensive step, do once per split)
            print("  Computing SIFT descriptors...")
            train_desc = compute_sift_descriptors(train_paths)
            val_desc = compute_sift_descriptors(val_paths)
            test_desc = compute_sift_descriptors(test_paths)

            # 4. Iterate Vocab Size
            for vocab in PARAMS_GRID["vocab_size"]:
                # Build Vocab on TRAIN only
                kmeans = build_vocabulary(train_desc, vocab)

                # Featurize
                X_train = compute_bow_features(train_desc, kmeans, vocab)
                X_val = compute_bow_features(val_desc, kmeans, vocab)
                X_test = compute_bow_features(test_desc, kmeans, vocab)

                # 5. Iterate Classifiers
                for clf_type in PARAMS_GRID["classifier"]:
                    # Specific params for clf
                    param_list = PARAMS_GRID["C"] if clf_type == "svm" else PARAMS_GRID["n_neighbors"]

                    for p in param_list:
                        run_id += 1
                        print(f"  Run {run_id}: Vocab={vocab}, {clf_type}={p}")

                        if clf_type == "svm":
                            clf = SVC(kernel='linear', C=p)
                        else:
                            clf = KNeighborsClassifier(n_neighbors=p)

                        # Train
                        clf.fit(X_train, train_y)

                        # Validate (Decision metric)
                        val_pred = clf.predict(X_val)
                        val_acc = accuracy_score(val_y, val_pred)

                        # Test (Reporting metric - strictly only for final table)
                        test_pred = clf.predict(X_test)
                        test_acc = accuracy_score(test_labels, test_pred)

                        writer.writerow([domain, run_id, vocab, clf_type, p, val_acc, test_acc])
                        f.flush()

                        # Save CM and Errors for the report
                        cm = confusion_matrix(test_labels, test_pred)
                        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
                        fig, ax = plt.subplots(figsize=(6, 6))
                        disp.plot(ax=ax, cmap="Purples", colorbar=False)
                        plt.title(f"{domain} {clf_type}({p}) V={vocab}")
                        plt.savefig(FIG_DIR / f"{domain}_run{run_id}_cm.png")
                        plt.close()

                        save_error_cases(domain, run_id, X_test, test_labels, test_pred, test_paths, classes)


if __name__ == "__main__":
    run_experiment_logic()