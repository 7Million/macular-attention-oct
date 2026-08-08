import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Explicit system imports
from models.resnet import get_resnet50
from models.convnext import get_convnext_tiny
from models.resnet_cbam import get_resnet50_cbam

def train_model(model_name="resnet50", data_dir="/root/keele/oct_data/archive (5)/OCT2017 ", num_epochs=10, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Training for: {model_name.upper()}")
    print(f"Using device: {device}")
    print(f"Target Dataset Directory: {data_dir}")

    # Explicitly use absolute repository root path for all results artifacts
    repo_root = "/root/keele/macular-attention-oct"
    base_results_path = os.path.join(repo_root, "results")
    tables_path = os.path.join(base_results_path, "tables")
    figures_path = os.path.join(base_results_path, "figures")
    
    os.makedirs(tables_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Standard ImageNet transform modifications
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    train_path = os.path.join(data_dir, "train")

    # Swapped validation tracking split path configuration to the robust 'test' directory.
    # The original 'val' split contains only 32 images (8 per class), which is too small 
    # and unbalanced for reliable evaluation. During baseline ResNet50 testing, this tiny
    # sample size caused extreme validation spikes—hitting 100% accuracy prematurely by 
    # epoch 2, followed by violent drops and severe overfitting that distorted metrics.
    # Using the larger 'test' split guarantees a stable, statistically robust validation monitor.
    val_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Could not locate training directory at: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Could not locate validation directory at: {val_path}")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset Loaded | Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    if model_name == "resnet50":
        model = get_resnet50(num_classes=4)
    elif model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=4)
    elif model_name == "resnet50_cbam":
        model = get_resnet50_cbam(num_classes=4)
    else:
        raise ValueError(f"Unsupported model string: {model_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Tracking structures for results compilation
    history = {"epoch": [], "train_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Log performance metrics per step
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_acc)

    # 1. Export performance tables (CSV metric tracking)
    df_metrics = pd.DataFrame(history)
    csv_filename = os.path.join(tables_path, f"{model_name}_training_metrics.csv")
    df_metrics.to_csv(csv_filename, index=False)
    print(f"Metrics table saved successfully to: {csv_filename}")

    # 2. Export performance plots (Matplotlib graphic tracking)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="blue")
    plt.title(f"{model_name.upper()} Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_accuracy"], label="Val Accuracy", color="green")
    plt.title(f"{model_name.upper()} Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(figures_path, f"{model_name}_curves.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Performance plots saved successfully to: {plot_filename}")

    # 3. Export raw model weight checkpoints
    save_name = os.path.join(base_results_path, f"{model_name}_baseline.pth")
    torch.save(model.state_dict(), save_name)
    print(f"Checkpoint successfully stored as: {save_name}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vision baseline models for Retinal OCT classification.")
    parser.add_argument("--model", type=str, default="all", choices=["resnet50", "resnet50_cbam", "convnext_tiny", "all"])
    parser.add_argument("--data_dir", type=str, default="/root/keele/oct_data/archive (5)/OCT2017 ")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()

    if args.model == "all":
        print("Running complete sequential benchmarking experiment...")
        train_model(model_name="resnet50", data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size)
        train_model(model_name="convnext_tiny", data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size)
        train_model(model_name="resnet50_cbam", data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size)
    else:
        train_model(model_name=args.model, data_dir=args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size)
