"""
cnn_resnet18_icub
"""

# ===== Imports =====
from pathlib import Path
import itertools
import time
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ===== Global configuration =====
DATA_ROOT = Path("/Users/delaynomore/Downloads/iCubWorld1.0")
DOMAINS = ["human", "robot"]

# Manual hyperparameter grid
HYPERPARAM_GRID = {
    "batch_size": [16, 32],
    "learning_rate": [1e-3, 5e-4],
    "weight_decay": [0.0, 1e-4],
    "num_epochs": [15],
}

RESULTS_CSV = "cnn_results_icub.csv"
FIG_DIR = Path("figures_cnn")
FIG_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# ===== Data loading =====
def create_transforms():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def create_datasets_and_loaders(domain, batch_size):
    train_transform, eval_transform = create_transforms()
    train_dir = DATA_ROOT / domain / "train"
    test_dir = DATA_ROOT / domain / "test"

    full_train_dataset_aug = datasets.ImageFolder(train_dir, transform=train_transform)
    full_train_dataset_eval = datasets.ImageFolder(train_dir, transform=eval_transform)  # For Validation

    num_train = len(full_train_dataset_aug)
    indices = np.random.permutation(num_train)
    split = int(0.8 * num_train)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = Subset(full_train_dataset_aug, train_indices)
    val_dataset = Subset(full_train_dataset_eval, val_indices)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_train_dataset_aug.classes


# ===== Model =====
def build_resnet18_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


# ===== Visualization Helper =====
def save_error_cases(model, data_loader, domain, run_id, class_names):

    model.eval()
    input_batches, prediction_batches, label_batches = [], [], []

    # Grab one batch or iterate until we find errors
    with torch.no_grad():
        for input_images, ground_truth_labels in data_loader:
            input_images = input_images.to(DEVICE)
            model_outputs = model(input_images)
            _, predicted_classes = torch.max(model_outputs, 1)

            # Store data to find errors
            input_batches.append(input_images.cpu())
            prediction_batches.append(predicted_classes.cpu())
            label_batches.append(ground_truth_labels.cpu())

            # Limit to checking first few batches to save time
            if len(input_batches) * data_loader.batch_size > 200:
                break

    all_inputs = torch.cat(input_batches)
    all_predictions = torch.cat(prediction_batches)
    all_labels = torch.cat(label_batches)

    # Find indices where prediction != label
    error_indices = (all_predictions != all_labels).nonzero(as_tuple=False).squeeze()

    if error_indices.numel() == 0:
        return  # No errors found

    # Denormalize for plotting
    inverse_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    # Plot top 5 errors
    num_errors_to_plot = min(5, error_indices.numel())
    fig, axes = plt.subplots(1, num_errors_to_plot, figsize=(15, 3))
    if num_errors_to_plot == 1: axes = [axes]

    for plot_index, error_index in enumerate(error_indices[:num_errors_to_plot]):
        image_tensor = all_inputs[error_index]
        image_array = inverse_normalize(image_tensor).permute(1, 2, 0).numpy()
        image_array = np.clip(image_array, 0, 1)

        true_class_name = class_names[all_labels[error_index].item()]
        predicted_class_name = class_names[all_predictions[error_index].item()]

        axes[plot_index].imshow(image_array)
        axes[plot_index].set_title(f"True: {true_class_name}\nPred: {predicted_class_name}", color='red', fontsize=10)
        axes[plot_index].axis('off')

    plt.suptitle(f"{domain} Run {run_id} - Error Cases")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_errors.png")
    plt.close()


# ===== Training =====
def train_one_model(domain, hyperparams, run_id):
    batch_size = hyperparams["batch_size"]
    learning_rate = hyperparams["learning_rate"]
    weight_decay = hyperparams["weight_decay"]
    num_epochs = hyperparams["num_epochs"]

    train_loader, val_loader, test_loader, class_names = create_datasets_and_loaders(domain, batch_size)
    model = build_resnet18_model(len(class_names))

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_validation_accuracy = 0.0
    training_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"[{domain}] Run {run_id} | Epoch {epoch + 1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                current_loader = train_loader
            else:
                model.eval()
                current_loader = val_loader

            cumulative_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for input_images, ground_truth_labels in current_loader:
                input_images, ground_truth_labels = input_images.to(DEVICE), ground_truth_labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    model_outputs = model(input_images)
                    _, predicted_classes = torch.max(model_outputs, 1)
                    loss = loss_criterion(model_outputs, ground_truth_labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                cumulative_loss += loss.item() * input_images.size(0)
                correct_predictions += torch.sum(predicted_classes == ground_truth_labels.data).item()
                total_samples += input_images.size(0)

            epoch_accuracy = correct_predictions / total_samples
            epoch_loss = cumulative_loss / total_samples

            training_history[f"{phase}_loss"].append(epoch_loss)
            training_history[f"{phase}_acc"].append(epoch_accuracy)

            if phase == "val" and epoch_accuracy > best_validation_accuracy:
                best_validation_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

        print(f"  Val Acc: {training_history['val_acc'][-1]:.4f}")

    # Load best model for testing
    model.load_state_dict(best_model_weights)
    plot_training_curves(domain, run_id, training_history)

    # Evaluate Test
    test_accuracy, confusion_mat, true_labels, predicted_labels = evaluate_on_test(model, test_loader)
    plot_confusion_matrix(domain, run_id, confusion_mat, class_names)


    save_error_cases(model, test_loader, domain, run_id, class_names)

    return best_validation_accuracy, test_accuracy


def evaluate_on_test(model, test_loader):
    model.eval()
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for input_images, ground_truth_labels in test_loader:
            input_images, ground_truth_labels = input_images.to(DEVICE), ground_truth_labels.to(DEVICE)
            model_outputs = model(input_images)
            _, predicted_classes = torch.max(model_outputs, 1)
            true_labels.extend(ground_truth_labels.cpu().tolist())
            predicted_labels.extend(predicted_classes.cpu().tolist())

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    return accuracy, confusion_mat, true_labels, predicted_labels


def plot_training_curves(domain, run_id, training_history):
    epoch_numbers = range(1, len(training_history["train_loss"]) + 1)
    # Loss
    plt.figure()
    plt.plot(epoch_numbers, training_history["train_loss"], label="Train")
    plt.plot(epoch_numbers, training_history["val_loss"], label="Val")
    plt.title(f"{domain} Run {run_id} Loss")
    plt.legend()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_loss.png")
    plt.close()
    # Accuracy
    plt.figure()
    plt.plot(epoch_numbers, training_history["train_acc"], label="Train")
    plt.plot(epoch_numbers, training_history["val_acc"], label="Val")
    plt.title(f"{domain} Run {run_id} Accuracy")
    plt.legend()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_acc.png")
    plt.close()


def plot_confusion_matrix(domain, run_id, confusion_mat, class_names):
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{domain} Run {run_id} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_cm.png")
    plt.close()


def run_experiments():
    with open(RESULTS_CSV, "w", newline="") as results_file:
        csv_writer = csv.writer(results_file)
        csv_writer.writerow(["domain", "run_id", "batch", "lr", "wd", "epochs", "best_val_acc", "test_acc"])

        run_id = 0
        for domain in DOMAINS:
            # Grid search
            for batch_size, learning_rate, weight_decay, num_epochs in itertools.product(
                    HYPERPARAM_GRID["batch_size"],
                    HYPERPARAM_GRID["learning_rate"],
                    HYPERPARAM_GRID["weight_decay"],
                    HYPERPARAM_GRID["num_epochs"]
            ):
                run_id += 1
                hyperparams = {"batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay, "num_epochs": num_epochs}
                print(f"--- Domain: {domain} | Run: {run_id} ---")

                validation_accuracy, test_accuracy = train_one_model(domain, hyperparams, run_id)

                csv_writer.writerow([domain, run_id, batch_size, learning_rate, weight_decay, num_epochs, validation_accuracy, test_accuracy])
                results_file.flush()


if __name__ == "__main__":
    run_experiments()