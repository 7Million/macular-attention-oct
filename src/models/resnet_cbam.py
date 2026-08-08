import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from models.attention import CBAM

def get_resnet50_cbam(num_classes=4):
    # Loading standard pre-trained weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Injectin CBAM modules into the end of major ResNet feature extraction stages
    # This forces the model to filter features at multiple scales
    model.layer1 = nn.Sequential(model.layer1, CBAM(channels=256))
    model.layer2 = nn.Sequential(model.layer2, CBAM(channels=512))
    model.layer3 = nn.Sequential(model.layer3, CBAM(channels=1024))
    model.layer4 = nn.Sequential(model.layer4, CBAM(channels=2048))
    
    # Adjusting the output head for 4 clinical categories
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
