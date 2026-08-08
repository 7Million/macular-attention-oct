import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

# =====================================================================
# CORE PATH CONFIGURATION
# =====================================================================
# After deployment on Hugging Face, REPO_ROOT becomes the active folder directory "."
# CUrrently running locally on RunPod, so it scales up relative to the file placement
REPO_ROOT = "." if os.path.exists("./results") else "/root/keele/macular-attention-oct"
BASE_RESULTS_PATH = os.path.join(REPO_ROOT, "results")
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Standard transformation pipeline matching training benchmarks
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# INTERNAL ATTENTION MODULE STRUCTURES
# =====================================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.channel_attention(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_attention(spatial)
        return x * spatial

# =====================================================================
# INTERFACE GRAD-CAM EXPLAINABILITY ENGINE
# =====================================================================
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
        probabilities = torch.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Dynamic dimension checker resolves standard vs. custom layer shapes
        if len(self.gradients.shape) == 4:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        else:
            # Safe boundary fallback for custom spatial attention channels
            weights = torch.mean(self.gradients, dim=-1, keepdim=True)
            
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.clamp(cam, min=0)
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        self.forward_hook.remove()
        self.backward_hook.remove()
        
        # Detach the active calculation graphs from RAM before converting to numpy
        return cam.cpu().numpy(), target_class, probabilities.detach().cpu().numpy()

# =====================================================================
# RUNTIME DETACHED ARCHITECTURE CONFIGURATION LOADER
# =====================================================================
def load_architecture(model_name):
    from torchvision.models import resnet50, convnext_tiny
    
    if model_name == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        target_layer = model.layer4[-1].conv3
    elif model_name == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        target_layer = model.features[-1][-1].block
    elif model_name == "resnet50_cbam":
        model = resnet50(weights=None)
        model.layer1 = nn.Sequential(model.layer1, CBAM(channels=256))
        model.layer2 = nn.Sequential(model.layer2, CBAM(channels=512))
        model.layer3 = nn.Sequential(model.layer3, CBAM(channels=1024))
        model.layer4 = nn.Sequential(model.layer4, CBAM(channels=2048))
        model.fc = nn.Linear(model.fc.in_features, 4)
        # Modifying hook to check the full composite sequence end layer block
        target_layer = model.layer4[-1]
    else:
        raise ValueError(f"Unknown architecture option selection: {model_name}")

    checkpoint_path = os.path.join(BASE_RESULTS_PATH, f"{model_name}_baseline.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing weights binary file. Ensure it sits at: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    return model, target_layer

# =====================================================================
# DIAGNOSTIC INFERENCE RUN PROCESSING PIPELINE
# =====================================================================
def predict_and_explain(input_image, model_selection):
    if input_image is None:
        return None, "Please upload an image."

    try:
        model, target_layer = load_architecture(model_selection)
        img_pil = Image.fromarray(input_image.astype('uint8'), 'RGB')
        input_tensor = INFERENCE_TRANSFORM(img_pil).unsqueeze(0)
        
        cam_engine = GradCAM(model, target_layer)
        heatmap, pred_idx, probabilities = cam_engine.generate_heatmap(input_tensor)
        
        h, w, _ = input_image.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        blended_output = cv2.addWeighted(input_image, 0.7, color_heatmap, 0.3, 0)
        confidences = {CLASS_NAMES[i]: float(probabilities[0][i]) for i in range(4)}
        
        return blended_output, confidences
    except Exception as e:
        return None, {f"Execution Error Breakdown: {str(e)}": 1.0}

# =====================================================================
# GRADIO USER INTERFACE DESIGN
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # MacularAttentionNet: Retinal OCT Diagnostic Space
        An interactive, clinical explainability dashboard built for dissertation research. Upload an optical coherence 
        tomography scan to analyze pathologies across standard legacy, hybrid attention, and modernized vision baselines.
        """
    )
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Retinal OCT Image (JPEG/PNG)", type="numpy")
            model_dropdown = gr.Dropdown(
                choices=["resnet50", "resnet50_cbam", "convnext_tiny"], 
                value="resnet50", 
                label="Select Architectural Model Evaluation Backbone"
            )
            submit_btn = gr.Button("Execute Diagnostic Inference Run", variant="primary")
        with gr.Column():
            output_heatmap = gr.Image(label="Grad-CAM Localization Heatmap Overlay")
            output_labels = gr.Label(label="Model Prediction Confidence Distributions", num_top_classes=4)

    submit_btn.click(
        fn=predict_and_explain, 
        inputs=[input_img, model_dropdown], 
        outputs=[output_heatmap, output_labels]
    )

if __name__ == "__main__":
    demo.launch()
