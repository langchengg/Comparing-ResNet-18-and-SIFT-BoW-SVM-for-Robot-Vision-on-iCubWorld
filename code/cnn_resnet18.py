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
def save_error_cases(model, loader, domain, run_id, class_names):

    model.eval()
    inputs_list, preds_list, labels_list = [], [], []

    # Grab one batch or iterate until we find errors
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store data to find errors
            inputs_list.append(inputs.cpu())
            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())

            # Limit to checking first few batches to save time
            if len(inputs_list) * loader.batch_size > 200:
                break

    inputs = torch.cat(inputs_list)
    preds = torch.cat(preds_list)
    labels = torch.cat(labels_list)

    # Find indices where prediction != label
    errors = (preds != labels).nonzero(as_tuple=False).squeeze()

    if errors.numel() == 0:
        return  # No errors found

    # Denormalize for plotting
    inv_normalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])

    # Plot top 5 errors
    num_to_plot = min(5, errors.numel())
    fig, axes = plt.subplots(1, num_to_plot, figsize=(15, 3))
    if num_to_plot == 1: axes = [axes]

    for i, idx in enumerate(errors[:num_to_plot]):
        img_tensor = inputs[idx]
        img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        true_cls = class_names[labels[idx].item()]
        pred_cls = class_names[preds[idx].item()]

        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}", color='red', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f"{domain} Run {run_id} - Error Cases")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_errors.png")
    plt.close()


# ===== Training =====
def train_one_model(domain, hyperparams, run_id):
    batch_size = hyperparams["batch_size"]
    lr = hyperparams["learning_rate"]
    wd = hyperparams["weight_decay"]
    num_epochs = hyperparams["num_epochs"]

    train_loader, val_loader, test_loader, class_names = create_datasets_and_loaders(domain, batch_size)
    model = build_resnet18_model(len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=wd)

    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"[{domain}] Run {run_id} | Epoch {epoch + 1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

            epoch_acc = running_corrects / total
            epoch_loss = running_loss / total

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

        print(f"  Val Acc: {history['val_acc'][-1]:.4f}")

    # Load best model for testing
    model.load_state_dict(best_wts)
    plot_training_curves(domain, run_id, history)

    # Evaluate Test
    test_acc, cm, y_true, y_pred = evaluate_on_test(model, test_loader)
    plot_confusion_matrix(domain, run_id, cm, class_names)


    save_error_cases(model, test_loader, domain, run_id, class_names)

    return best_val_acc, test_acc


def evaluate_on_test(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return acc, cm, y_true, y_pred


def plot_training_curves(domain, run_id, history):
    epochs = range(1, len(history["train_loss"]) + 1)
    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.title(f"{domain} Run {run_id} Loss")
    plt.legend()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_loss.png")
    plt.close()
    # Acc
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.title(f"{domain} Run {run_id} Accuracy")
    plt.legend()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_acc.png")
    plt.close()


def plot_confusion_matrix(domain, run_id, cm, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{domain} Run {run_id} CM")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{domain}_run{run_id}_cm.png")
    plt.close()


def run_experiments():
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "run_id", "batch", "lr", "wd", "epochs", "best_val_acc", "test_acc"])

        run_id = 0
        for domain in DOMAINS:
            # Grid search
            for bs, lr, wd, ep in itertools.product(
                    HYPERPARAM_GRID["batch_size"],
                    HYPERPARAM_GRID["learning_rate"],
                    HYPERPARAM_GRID["weight_decay"],
                    HYPERPARAM_GRID["num_epochs"]
            ):
                run_id += 1
                params = {"batch_size": bs, "learning_rate": lr, "weight_decay": wd, "num_epochs": ep}
                print(f"--- Domain: {domain} | Run: {run_id} ---")

                val_acc, test_acc = train_one_model(domain, params, run_id)

                writer.writerow([domain, run_id, bs, lr, wd, ep, val_acc, test_acc])
                f.flush()


if __name__ == "__main__":
    run_experiments()