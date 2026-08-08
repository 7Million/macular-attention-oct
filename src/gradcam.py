import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import argparse

from models.resnet import get_resnet50
from models.convnext import get_convnext_tiny

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0)
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        self.forward_hook.remove()
        self.backward_hook.remove()
        
        return cam.cpu().numpy(), target_class

def generate_and_save_heatmap(image_path, model_name="resnet50"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    repo_root = "/root/keele/macular-attention-oct"
    base_results_path = os.path.join(repo_root, "results")
    heatmaps_path = os.path.join(base_results_path, "heatmaps")
    os.makedirs(heatmaps_path, exist_ok=True)

    class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found at path: {image_path}")

    img_pil = Image.open(image_path)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Safely route hook tracking directly to the target layer profiles
    if model_name == "resnet50":
        model = get_resnet50(num_classes=4)
        checkpoint_path = os.path.join(base_results_path, f"{model_name}_baseline.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        target_layer = model.layer4[-1].conv3
        
    elif model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=4)
        checkpoint_path = os.path.join(base_results_path, f"{model_name}_baseline.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        target_layer = model.features[-1][-1].block
        
    elif model_name == "resnet50_cbam":
        from models.resnet_cbam import get_resnet50_cbam
        model = get_resnet50_cbam(num_classes=4)
        checkpoint_path = os.path.join(base_results_path, f"{model_name}_baseline.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        # Hooks directly into my engineered spatial attention convolutional block layer
        target_layer = model.layer4[-1].spatial_attention
    else:
        raise ValueError(f"Unsupported model string parameter: {model_name}")

    model = model.to(device)
    
    cam_engine = GradCAM(model, target_layer)
    heatmap, pred_idx = cam_engine.generate_heatmap(input_tensor)
    predicted_class_name = class_names[pred_idx]

    # Process and blend arrays using openCV
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    blended_img = cv2.addWeighted(orig_img, 0.7, color_heatmap, 0.3, 0)

    # Rendering a visualization dashboard
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Target the first subplot axis window panel [0]
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Retinal OCT Scan")
    axes[0].axis("off")
    
    # Target the second subplot axis window panel [1], hope this works
    axes[1].imshow(blended_img)
    axes[1].set_title(f"{model_name.upper()} Localization Layer\nPredicted Diagnosis: {predicted_class_name}")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = os.path.join(heatmaps_path, f"{base_name}_{model_name}_gradcam.png")
    
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Grad-CAM heatmap saved successfully to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual explainability layers for trained OCT checkpoints.")
    parser.add_argument("--image_path", type=str, required=True, help="Absolute path to sample scan file.")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "convnext_tiny", "resnet50_cbam"])
    
    args = parser.parse_args()
    generate_and_save_heatmap(image_path=args.image_path, model_name=args.model)
