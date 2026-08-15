# =====================================================================
# MacularAttentionNet - Gradio App (Final Version)
# Hugging Face Repo: https://huggingface.co/jeepaz/macular-attention-oct
# =====================================================================

import cv2
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms, models
from huggingface_hub import hf_hub_download

# ====================== CONFIG ======================
REPO_ID = "jeepaz/macular-attention-oct"
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================== ATTENTION MODULES ======================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
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
        super().__init__()
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


# ====================== MODEL BUILDERS ======================
def get_resnet50(num_classes=4):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_convnext_tiny(num_classes=4):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def get_resnet50_cbam(num_classes=4):
    """
    ResNet50 with CBAM attached after each residual stage.
    This must match the architecture used during training.
    """
    model = models.resnet50(weights=None)

    # Attach CBAM after each residual stage
    model.layer1 = nn.Sequential(model.layer1, CBAM(256))
    model.layer2 = nn.Sequential(model.layer2, CBAM(512))
    model.layer3 = nn.Sequential(model.layer3, CBAM(1024))
    model.layer4 = nn.Sequential(model.layer4, CBAM(2048))

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ====================== GRAD-CAM ======================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # Clean up hooks
        self.forward_hook.remove()
        self.backward_hook.remove()

        return cam.cpu().numpy(), target_class, probs[0].detach().cpu().numpy()


# ====================== LOAD MODEL ======================
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "resnet50":
        model = get_resnet50(num_classes=4)
        target_layer = model.layer4[-1].conv3

    elif model_name == "convnext_tiny":
        model = get_convnext_tiny(num_classes=4)
        target_layer = model.features[-1][-1].block

    elif model_name == "resnet50_cbam":
        model = get_resnet50_cbam(num_classes=4)
        target_layer = model.layer4[-1].channel_attention
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Download weights from Hugging Face
    weight_files = {
        "resnet50": "resnet50_baseline.pth",
        "convnext_tiny": "convnext_tiny_baseline.pth",
        "resnet50_cbam": "resnet50_cbam_baseline.pth"
    }

    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=weight_files[model_name]
    )

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Remove DataParallel "module." prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, target_layer, device


# ====================== PREDICTION FUNCTION ======================
def predict(image, model_name):
    if image is None:
        return None, {"Please upload an image": 1.0}

    try:
        model, target_layer, device = load_model(model_name)

        # Prepare image
        img_pil = Image.fromarray(image.astype("uint8")).convert("RGB")
        input_tensor = INFERENCE_TRANSFORM(img_pil).unsqueeze(0).to(device)

        # Generate Grad-CAM
        cam_engine = GradCAM(model, target_layer)
        heatmap, pred_idx, probs = cam_engine.generate(input_tensor)

        # Create overlay
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(image, 0.65, color_heatmap, 0.35, 0)

        confidences = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(4)
        }

        return overlay, confidences

    except Exception as e:
        return None, {f"Error: {str(e)}": 1.0}


# ====================== GRADIO INTERFACE ======================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # MacularAttentionNet: Retinal OCT Diagnostic Space
        Upload an OCT scan to classify macular disease and visualise model attention using Grad-CAM.
        """
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="Upload Retinal OCT Image (JPEG/PNG)",
                type="numpy"
            )
            model_dropdown = gr.Dropdown(
                choices=["resnet50", "resnet50_cbam", "convnext_tiny"],
                value="resnet50",
                label="Select Model"
            )
            submit_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column():
            output_heatmap = gr.Image(label="Grad-CAM Heatmap Overlay")
            output_labels = gr.Label(label="Prediction Confidence", num_top_classes=4)

    submit_btn.click(
        fn=predict,
        inputs=[input_img, model_dropdown],
        outputs=[output_heatmap, output_labels]
    )

if __name__ == "__main__":
    demo.launch()