# Functions to compute attention

def compute_attention_rollout(attn_weights, discard_ratio=0.0):

    # Start with identity matrix 
    device = attn_weights[0].device
    B, _, N, _ = attn_weights[0].shape
    result = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

    for layer_attn in attn_weights:
        # Mean over heads: [B, Heads, N, N] → [B, N, N]
        attn_heads_fused = layer_attn.mean(dim=1)

        # (Optional) Discard lowest attention values
        if discard_ratio > 0:
            flat = attn_heads_fused.view(B, -1)
            num_entries = flat.size(1)
            num_discard = int(num_entries * discard_ratio)

            # Mask out lowest values per sample
            threshold, _ = torch.kthvalue(flat, num_discard, dim=1, keepdim=True)
            mask = flat >= threshold
            flat = flat * mask
            attn_heads_fused = flat.view(B, N, N)

        attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)

        result = torch.bmm(attn_heads_fused, result)

    return result 

def visualise_random_attention_maps(dataset, model, num_images=4, transform=None, title_prefix=""):
    model.eval()
    indices = random.sample(range(len(dataset)), num_images)

    for idx in indices:
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)

        with torch.no_grad():
            _, attn_weights = model.forward_encoder_with_attention(input_tensor)
            rollout = compute_attention_rollout(attn_weights)

            # CLS-to-patch attention
            cls_attention = rollout[:, 0, 1:]
            num_patches = cls_attention.shape[-1]
            side = int(num_patches ** 0.5)
            attn_map = cls_attention.view(1, side, side).cpu().numpy()

            # Resize to image size
            import cv2
            attn_resized = cv2.resize(attn_map[0], (image_tensor.shape[2], image_tensor.shape[1]))

        if transform:
            image_to_plot = transform(image_tensor.permute(1, 2, 0).cpu().numpy())
        else:
            image_to_plot = image_tensor.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.imshow(image_to_plot.squeeze(), cmap='gray')
        plt.imshow(attn_resized, cmap='jet', alpha=0.5)
        plt.title(f"{title_prefix}Label: {label} | Sample #{idx}")
        plt.axis('off')
        plt.show()

# Loading weights

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import optuna
from vitmae_encoder_attention import mae_vit_encoder2 
from functools import partial
from tqdm import tqdm
import logging
import random

from google.colab import drive
drive.mount('/content/drive')

# ---------------------------- #
#         CONFIGURATION
# ---------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Directories --------- #
data_dir = '/content/drive/My Drive/ultra-widefield_images_soft'
feature_save_dir = '/content/drive/My Drive/ultra-widefield_images_soft/featuresv9-'

# Make sure the feature directory exists
os.makedirs(feature_save_dir, exist_ok=True)

# --------- Pretrained weights --------- #
pretrained_weights = '/content/drive/My Drive/checkpoint_vitmae_100000_ps64_bs32_la_round.pth'

# --------- Logging --------- #
log_file = os.path.join(feature_save_dir, 'feature_extraction.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ---------------------------- #
#   DATA TRANSFORM AND LOADER
# ---------------------------- #

basic_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')
test_dir  = os.path.join(data_dir, 'test')

train_dataset = ImageFolder(train_dir, transform=basic_transform)
val_dataset = ImageFolder(val_dir, transform=basic_transform)
test_dataset = ImageFolder(test_dir, transform=basic_transform)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=16, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=16, shuffle=False)
}

datasets_dict = {
    'train': train_dataset,
    'val': val_dataset,
    'test': test_dataset
}

# ---------------------------- #
#      LOAD PRETRAINED MAE
# ---------------------------- #
pretrained_model = mae_vit_encoder2(in_chans=3).to(device).eval()

checkpoint = torch.load(pretrained_weights, map_location=device)

print(checkpoint.keys())

pretrained_dict = {
    k: v for k, v in checkpoint['model_state_dict'].items()
    if k in pretrained_model.state_dict()
}

try:
    pretrained_model.load_state_dict(pretrained_dict)
    logging.info("Pretrained weights loaded successfully.")
    print("Pretrained weights loaded successfully.")
except RuntimeError as e:
    logging.error(f"Error loading state_dict: {e}")
    print(f"Error loading state_dict: {e}")
    exit(1)

# ---------------------------- #
#    FEATURE EXTRACTION
# ---------------------------- #
def extract_and_save_features(dataloader, phase):
    phase_dir = os.path.join(feature_save_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)
    logging.info(f"Processing phase: {phase}")
    print(f"Processing phase: {phase}")

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            B = inputs.size(0)

            print(f"Processing batch {idx + 1}/{len(dataloader)} with batch size {B}")
            logging.info(f"Processing batch {idx + 1}/{len(dataloader)} with batch size {B}")

            # Forward pass through MAE encoder
            try:
                latent = pretrained_model.forward_encoder(inputs)
            except Exception as e:
                logging.error(f"Error during forward pass: {e}")
                print(f"Error during forward pass: {e}")
                continue

            # Extract CLS token (first token)
            cls_features = latent[:, 0, :]

            # Save features and labels
            for b in range(B):
                feature_path = os.path.join(phase_dir, f'features_{idx * dataloader.batch_size + b}.pt')
                try:
                    torch.save({'features': cls_features[b].cpu(), 'label': labels[b].cpu()}, feature_path)
                    logging.info(f"Saved {feature_path}")
                    print(f"Saved {feature_path}")
                except Exception as e:
                    logging.error(f"Error saving {feature_path}: {e}")
                    print(f"Error saving {feature_path}: {e}")

            logging.info(f"Batch {idx + 1}/{len(dataloader)} processed successfully")
            print(f"Batch {idx + 1}/{len(dataloader)} processed successfully")

def run_feature_extraction():
    print("Starting feature extraction ...")
    logging.info("Starting feature extraction ...")
    phases = ['train', 'val', 'test']
    for phase in phases:
        print(f"Extracting features for {phase} set...")
        logging.info(f"Extracting features for {phase} set...")
        extract_and_save_features(dataloaders[phase], phase)
    print("Feature extraction complete")
    logging.info("Feature extraction complete")

# Show attention for 3 val and 3 train images

# De-normalise for visualisation
def denormalise(x):
    return x * 0.5 + 0.5 

# View 3 random training images
visualise_random_attention_maps(
    dataset=train_dataset,
    model=pretrained_model,
    num_images=3,
    transform=denormalise,
    title_prefix="Train – "
)

# View 3 random validation images
visualise_random_attention_maps(
    dataset=val_dataset,
    model=pretrained_model,
    num_images=3,
    transform=denormalise,
    title_prefix="Val – "
)
