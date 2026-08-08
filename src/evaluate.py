import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from models.resnet import get_resnet50
from models.convnext import get_convnext_tiny

def evaluate_model(model_name="resnet50", data_dir="/root/keele/oct_data/archive (5)/OCT2017 "):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing Post-Training Evaluation for: {model_name.upper()}")
    print(f"Using device: {device}")

    repo_root = "/root/keele/macular-attention-oct"
    base_results_path = os.path.join(repo_root, "results")
    tables_path = os.path.join(base_results_path, "tables")
    figures_path = os.path.join(base_results_path, "figures")

    os.makedirs(tables_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Transform parameters matching training configurations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    test_path = os.path.join(data_dir, "test")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not locate testing split directory at: {test_path}")

    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    class_names = test_dataset.classes 

    print(f"Testing Dataset Loaded | Total Balanced Samples: {len(test_dataset)}")

    # Handle dynamic structure loading across all my 3 experimental models
    if model_name == "resnet50":
        model = get_resnet50(num_classes=4)
    elif model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=4)
    elif model_name == "resnet50_cbam":
        from models.resnet_cbam import get_resnet50_cbam
        model = get_resnet50_cbam(num_classes=4)
    else:
        raise ValueError(f"Unsupported model string parameter: {model_name}")

    # Load the specific checkpoint brain file
    checkpoint_path = os.path.join(base_results_path, f"{model_name}_baseline.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing weight checkpoint target. Verify file sits at: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation inference pipeline across test array...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute and save classification reports
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    report_filename = os.path.join(tables_path, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_filename, index=True)
    print(f"Classification report (F1-Scores) exported cleanly to: {report_filename}")

    print("\n--- FINAL TEST PERFORMANCE METRICS ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 2. Compute and save confusion matrix figures
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    
    plt.title(f"Confusion Matrix Evaluation - {model_name.upper()}")
    plt.tight_layout()
    
    matrix_filename = os.path.join(figures_path, f"{model_name}_confusion_matrix.png")
    plt.savefig(matrix_filename, dpi=300)
    plt.close()
    print(f"Confusion Matrix graphic exported cleanly to: {matrix_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vision checkpoints for Retinal OCT classification.")
    parser.add_argument("--model", type=str, default="all", choices=["resnet50", "convnext_tiny", "resnet50_cbam", "all"])
    parser.add_argument("--data_dir", type=str, default="/root/keele/oct_data/archive (5)/OCT2017 ")
    
    args = parser.parse_args()

    if args.model == "all":
        print("Running complete sequential evaluation benchmark...")
        evaluate_model(model_name="resnet50", data_dir=args.data_dir)
        evaluate_model(model_name="convnext_tiny", data_dir=args.data_dir)
        evaluate_model(model_name="resnet50_cbam", data_dir=args.data_dir)
    else:
        evaluate_model(model_name=args.model, data_dir=args.data_dir)
