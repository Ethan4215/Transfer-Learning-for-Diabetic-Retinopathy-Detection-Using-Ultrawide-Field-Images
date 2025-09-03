# Masters Thesis Repository – DR Detection with XAI

This repository contains Python scripts used in the experiments for my MSc thesis on **Transfer Learning for Diabetic Retinopathy Detection Using Ultrawide-Field (UWF) Images**. The focus is on model interpretability using a variety of **explainable AI (XAI) techniques**.

## Files

- **Attention_Rollout.py**  
  Runs attention rollout visualisations for the Vision Transformer (ViT-MAE) model.  
  Requires the `vitmae_encoder_attention.py` module.

- **vitmae_encoder_attention.py**  
  Modified MAE encoder with attention hooks.  
  Used as a dependency for `Attention_Rollout.py`.

- **FM_Grad-CAM.py**  
  Applies Grad-CAM to the foundation model (ViT backbone + MLP head).

- **FM_Saliency.py**  
  Generates saliency maps for the foundation model.

- **ResNet_Grad-CAM.py**  
  Applies Grad-CAM to the ResNet-18 baseline model.

- **ResNet_Saliency.py**  
  Generates saliency maps for the ResNet-18 baseline model.

- **TSNE-UMAP.py**  
  Performs feature space analysis using t-SNE and UMAP to visualise model embeddings.

## Requirements

- Python 3.9+  
- PyTorch  
- torchvision  
- matplotlib  
- scikit-learn  
- umap-learn  

## Usage

Each script can be run independently. Example:

```bash
python FM_Saliency.py
