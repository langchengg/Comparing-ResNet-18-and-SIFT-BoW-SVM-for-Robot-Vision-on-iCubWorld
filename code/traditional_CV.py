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
    sift_extractor = cv2.SIFT_create(nfeatures=max_features)
    descriptor_list = []
    for image_path in image_paths:
        grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if grayscale_image is None:
            descriptor_list.append(None)
            continue
        _, descriptors = sift_extractor.detectAndCompute(grayscale_image, None)
        descriptor_list.append(descriptors)
    return descriptor_list


def build_vocabulary(descriptors_list, vocabulary_size, max_samples=100000):
    valid_descriptors = [descriptor for descriptor in descriptors_list if descriptor is not None]
    if not valid_descriptors: raise RuntimeError("No descriptors found.")

    # Stack and Force Float64 for Scikit-Learn compatibility
    all_descriptors = np.vstack(valid_descriptors).astype(np.float64)

    if len(all_descriptors) > max_samples:
        sample_indices = np.random.choice(len(all_descriptors), max_samples, replace=False)
        all_descriptors = all_descriptors[sample_indices]

    print(f"  Clustering {len(all_descriptors)} descriptors -> {vocabulary_size} words...")
    kmeans_model = MiniBatchKMeans(n_clusters=vocabulary_size, batch_size=1000, random_state=42, n_init='auto')
    kmeans_model.fit(all_descriptors)
    return kmeans_model


def compute_bow_features(descriptor_list, kmeans_model, vocabulary_size):
    bow_features = []
    for descriptor in descriptor_list:
        if descriptor is None:
            bow_features.append(np.zeros(vocabulary_size))
            continue

        # FIX: Explicit cast to float64 before prediction
        visual_words = kmeans_model.predict(descriptor.astype(np.float64))

        histogram, _ = np.histogram(visual_words, bins=np.arange(vocabulary_size + 1))
        histogram_norm = np.linalg.norm(histogram)
        if histogram_norm > 0: histogram = histogram / histogram_norm  # L2 norm usually better for SVM
        bow_features.append(histogram)
    return np.stack(bow_features)


# ===== Visualization =====
def save_error_cases(domain, run_id, bow_features_test, ground_truth_labels, predicted_labels, test_image_paths, class_names):

    error_indices = np.where(ground_truth_labels != predicted_labels)[0]
    if len(error_indices) == 0: return

    num_errors_to_plot = min(5, len(error_indices))
    fig, axes = plt.subplots(1, num_errors_to_plot, figsize=(15, 3))
    if num_errors_to_plot == 1: axes = [axes]

    for plot_index, error_index in enumerate(error_indices[:num_errors_to_plot]):
        image_path = test_image_paths[error_index]
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            true_class_name = class_names[ground_truth_labels[error_index]]
            predicted_class_name = class_names[predicted_labels[error_index]]

            axes[plot_index].imshow(image)
            axes[plot_index].set_title(f"T: {true_class_name}\nP: {predicted_class_name}", color='red', fontsize=10)
            axes[plot_index].axis('off')

    plt.suptitle(f"{domain} Run {run_id} Errors")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_errors.png")
    plt.close()


# ===== Experiment Logic =====
def run_bow_sift_experiments():
    with open(RESULTS_CSV, "w", newline="") as results_file:
        csv_writer = csv.writer(results_file)
        csv_writer.writerow(["domain", "run_id", "vocab", "clf_type", "param", "val_acc", "test_acc"])

        run_id = 0
        for domain in DOMAINS:
            print(f"\nProcessing Domain: {domain}")

            # 1. Load Data
            train_full_paths, train_full_labels, class_names = get_image_paths_and_labels(DATA_ROOT / domain / "train")
            test_image_paths, test_labels, _ = get_image_paths_and_labels(DATA_ROOT / domain / "test")

            # 2. Split Train into Train/Val (Essential for hyperparam tuning)
            train_image_paths, validation_image_paths, train_labels, validation_labels = train_test_split(
                train_full_paths, train_full_labels, test_size=0.2, stratify=train_full_labels, random_state=42
            )

            # 3. Compute Descriptors (expensive step, do once per split)
            print("  Computing SIFT descriptors...")
            train_descriptors = compute_sift_descriptors(train_image_paths)
            validation_descriptors = compute_sift_descriptors(validation_image_paths)
            test_descriptors = compute_sift_descriptors(test_image_paths)

            # 4. Iterate Vocab Size
            for vocabulary_size in PARAMS_GRID["vocab_size"]:
                # Build Vocab on TRAIN only
                kmeans_model = build_vocabulary(train_descriptors, vocabulary_size)

                # Featurize
                train_bow_features = compute_bow_features(train_descriptors, kmeans_model, vocabulary_size)
                validation_bow_features = compute_bow_features(validation_descriptors, kmeans_model, vocabulary_size)
                test_bow_features = compute_bow_features(test_descriptors, kmeans_model, vocabulary_size)

                # 5. Iterate Classifiers
                for classifier_type in PARAMS_GRID["classifier"]:
                    # Specific params for classifier
                    classifier_param_list = PARAMS_GRID["C"] if classifier_type == "svm" else PARAMS_GRID["n_neighbors"]

                    for classifier_param in classifier_param_list:
                        run_id += 1
                        print(f"  Run {run_id}: Vocab={vocabulary_size}, {classifier_type}={classifier_param}")

                        if classifier_type == "svm":
                            classifier = SVC(kernel='linear', C=classifier_param)
                        else:
                            classifier = KNeighborsClassifier(n_neighbors=classifier_param)

                        # Train
                        classifier.fit(train_bow_features, train_labels)

                        # Validate (Decision metric)
                        validation_predictions = classifier.predict(validation_bow_features)
                        validation_accuracy = accuracy_score(validation_labels, validation_predictions)

                        # Test (Reporting metric - strictly only for final table)
                        test_predictions = classifier.predict(test_bow_features)
                        test_accuracy = accuracy_score(test_labels, test_predictions)

                        csv_writer.writerow([domain, run_id, vocabulary_size, classifier_type, classifier_param, validation_accuracy, test_accuracy])
                        results_file.flush()

                        # Save CM and Errors for the report
                        confusion_mat = confusion_matrix(test_labels, test_predictions)
                        display = ConfusionMatrixDisplay(confusion_mat, display_labels=class_names)
                        fig, ax = plt.subplots(figsize=(6, 6))
                        display.plot(ax=ax, cmap="Purples", colorbar=False)
                        plt.title(f"{domain} {classifier_type}({classifier_param}) V={vocabulary_size}")
                        plt.savefig(FIG_DIR / f"{domain}_run{run_id}_cm.png")
                        plt.close()

                        save_error_cases(domain, run_id, test_bow_features, test_labels, test_predictions, test_image_paths, class_names)


if __name__ == "__main__":
    run_bow_sift_experiments()