# Macular Attention OCT

Attention-enhanced Convolutional Neural Networks for macular disease classification using Optical Coherence Tomography (OCT) images. This project was developed as part of an **MSc Computer Science dissertation at Keele University**.

#### Project Overview
The goal of this research is to improve the accuracy and explainability of deep learning models for classifying macular diseases from OCT scans. The models classify images into four categories:
- **CNV** (Choroidal Neovascularization)
- **DME** (Diabetic Macular Edema)
- **DRUSEN**
- **NORMAL**

---

### Models Implemented
- **ResNet50 (Baseline)**: Standard residual block baseline model.
- **ConvNeXt-Tiny (Baseline)**: Modernised convolutional baseline.
- **MacularAttentionNet (ResNet50 + CBAM)**: The primary custom architecture. It integrates a Convolutional Block Attention Module (CBAM) inside the final layer block, explicitly targeting the `channel_attention` block to spotlight localized pathological lesions (like deep intra-retinal fluid pockets or drusen deposits).

---

### Repository Structure
- `src/models/` → Custom neural network layouts (`resnet.py`, `resnet_cbam.py`, `convnext.py`).
- `src/train.py` → Deep learning training lifecycle framework.
- `src/gradcam.py` → Explainability engine utilizing locally saved model weights.
- `src/gradcam_hf.py` → Cloud-linked explainability engine utilizing Hugging Face Hub hosted checkpoints from my repository.
- `app/` → Gradio interactive web deployment demo interface.
- `results/heatmaps/` → Superimposed visual explainability dashboards showing original scans side-by-side with localized activation charts.

---

### Explainability & Qualitative Evaluation (Grad-CAM)

This framework includes a specialized **Grad-CAM (Gradient-weighted Class Activation Mapping)** engine to visually prove my dissertation's core contribution: showing that while standard baselines struggle with loose visual tracking boundary clouds, the attention-enhanced variant locks perfectly onto core retinal tissue features.

#### Setup Requirements
Make sure your environment can stream files directly from remote repositories:
```bash
pip install -r requirements.txt
```

#### 1. Running the Local Script (`src/gradcam.py`)

##### Prerequisites:
- Use this execution route if you are working within the workspace container and evaluating models using local `.pth` weight checkpoints stored inside `results/`.
- Start by downloading the `.pth` files from Hugging Face repo linked below, then upload into the right location, in my case I used `/root/keele/macular-attention-oct`. 
- Update every occurence of the above `repo_root` on all the scripts, Jupyter notebooks etc, to reflect the correct location.
- YOu need the linked dataset as well for the following commands to work. 

```bash
# Evaluate baseline ResNet50
python src/gradcam.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/CNV/CNV-1016042-1.jpeg" --model resnet50

# Evaluate baseline ConvNeXt-Tiny
python src/gradcam.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/CNV/CNV-1016042-1.jpeg" --model convnext_tiny

# Evaluate custom Attention model (CBAM)
python src/gradcam.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/CNV/CNV-1016042-1.jpeg" --model resnet50_cbam
```

#### 2. Running the Hugging Face Deployment Script (`src/gradcam_hf.py`)
##### Prerequisites:
- Dataset of any OCT scan images.

Use this command matrix to test the explainability dashboard by downloading model parameters seamlessly from my remote Hugging Face repository layer (`jeepaz/macular-attention-oct`). It maps structural layouts and handles backward activation hooks automatically.

##### Pathological Class Evaluation Commands:

*   **Choroidal Neovascularization (CNV)**
    ```bash
    python src/gradcam_hf.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/CNV/CNV-1016042-1.jpeg" --model resnet50_cbam
    ```
*   **Diabetic Macular Edema (DME)**
    ```bash
    python src/gradcam_hf.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/DME/DME-1102486-2.jpeg" --model resnet50_cbam
    ```
*   **Drusen**
    ```bash
    python src/gradcam_hf.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/DRUSEN/DRUSEN-1001621-1.jpeg" --model resnet50_cbam
    ```
*   **Normal (Control Frames)**
    ```bash
    python src/gradcam_hf.py --image_path "/workspace/oct_data/archive (5)/OCT2017 /test/NORMAL/NORMAL-1016061-1.jpeg" --model resnet50_cbam
    ```

*Note: All output visualization files are rendered at a crisp 300 DPI layout and saved under `results/heatmaps/` as `<image_name>_<model>_gradcam.png`.*

---

### Dataset
This project uses the publicly available **Kermany 2018 OCT** [dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018?select=OCT2017+).

---

### Let's connect
Engineered together by **Tadiwanashe Mataruse**.
🔗 [LinkedIn](https://www.linkedin.com/in/tadiwanashe-mataruse-73a3545b/) | 🤗 [Hugging Face](https://huggingface.co/jeepaz) | 🥞 [Stack Overflow](https://stackoverflow.com/users/2270348/tadiwanashe) | 📺 [YouTube](https://www.youtube.com/@HueyMataruse)

---

### License
This project is intended for educational and research purposes.
