# Needed Setup

import numpy as np
np.float = float

# Ensure optuna is installed
try:
    import optuna
    print("Optuna is already installed.")
except ImportError:
    print("Optuna not found. Installing...")
    !pip install optuna
    try:
        import optuna
        print("Optuna installed successfully.")
    except ImportError:
        print("Error: Optuna installation failed.")

import sys
import os

# Create the util directory if it doesn't exist
if not os.path.exists('/content/util'):
    os.makedirs('/content/util')
    print("Created directory: /content/util")
else:
    print("Directory /content/util already exists.")

# Now attempt to move the file
!mv pos_embed.py /content/util/ # Assuming pos_embed.py is in /content

# Make sure the current working directory includes util/
print("Current files:", os.listdir('/content'))
print("util folder contents:", os.listdir('/content/util'))

# Add /content to the Python path if it's not already
if '/content' not in sys.path:
    sys.path.append('/content')

from util.pos_embed import get_2d_sincos_pos_embed

# This code produces Grad-CAM

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
from model_vitmae_encoder import mae_vit_encoder  # Make sure this matches your local import path
from functools import partial
from tqdm import tqdm
import logging
import random
import cv2

from google.colab import drive
drive.mount('/content/drive')

# ---------------------------- #
#         CONFIGURATION
# ---------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Directories --------- #
data_dir = '/content/drive/My Drive/ultra-widefield_images_soft'
feature_save_dir = '/content/drive/My Drive/ultra-widefield_images_soft/featuresv8'

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


# Keep validation and test the same (no augmentation)
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
pretrained_model = mae_vit_encoder(in_chans=3).to(device).eval()

checkpoint = torch.load(pretrained_weights, map_location=device)

print(checkpoint.keys())

# Your original code pattern
# Corrected version
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

def get_gradcam_mae(model, input_tensor, target_class_idx):
    """
    Grad-CAM for MAE + MLP model - Version 4 using patch embeddings
    """
    model.eval()

    input_tensor.requires_grad_(True)
    model.zero_grad()

    # Forward pass
    output = model(input_tensor)
    class_score = output[0, target_class_idx]

    # Backward pass to get input gradients
    class_score.backward(retain_graph=True)

    # Get gradients with respect to input
    input_grads = input_tensor.grad.data  

    print(f"Input gradients shape: {input_grads.shape}")
    print(f"Input gradients range: {input_grads.min():.6f} to {input_grads.max():.6f}")

    if input_grads.abs().max() == 0:
        print("No gradients flowing to input!")
        return None

    # Convert input gradients to patch-level importance
    batch_size, channels, height, width = input_grads.shape

    patch_size = 64  
    cams = []
    for i in range(batch_size):
        # Sum gradients across channels
        grad_map = input_grads[i].abs().sum(dim=0)  

        # Divide into patches and average
        patches_h = height // patch_size  
        patches_w = width // patch_size 

        cam = np.zeros((patches_h, patches_w))

        for ph in range(patches_h):
            for pw in range(patches_w):
                patch_grad = grad_map[
                    ph*patch_size:(ph+1)*patch_size,
                    pw*patch_size:(pw+1)*patch_size
                ]
                cam[ph, pw] = patch_grad.mean().item()

        print(f"Patch-level CAM range: {cam.min():.6f} to {cam.max():.6f}")

        # Normalise
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        cams.append(cam)

    return np.array(cams)

def generate_gradcam_mae_visualization(model, dataloader, class_names, save_dir='gradcam_mae_maps'):
    """Generate Grad-CAM visualizations for MAE + MLP model"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    sample_count = 0
    max_samples = 5

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if sample_count >= max_samples:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

        for i in range(min(3, inputs.size(0))):
            if sample_count >= max_samples:
                break

            single_input = inputs[i:i+1].clone()
            pred_class = predicted[i].item()
            actual_class = labels[i].item()

            # Generate Grad-CAM
            cam = get_gradcam_mae(model, single_input, target_class_idx=pred_class)

            if cam is None:
                print(f"Failed to generate CAM for image {i}")
                continue

            cam = cam[0]  

            input_height, input_width = single_input.shape[2], single_input.shape[3]
            cam_resized = cv2.resize(cam, (input_width, input_height))

            original_img = single_input[0].detach().cpu()  
            original_img = original_img.permute(1, 2, 0) 

            original_img = (original_img + 1) / 2
            original_img = original_img.clamp(0, 1)

            original_img_gray = original_img.mean(dim=2).numpy()

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            ax[0].imshow(original_img_gray, cmap='gray')
            ax[0].set_title(f"Original\nActual: {class_names[actual_class]}")
            ax[0].axis('off')

            ax[1].imshow(cam_resized, cmap='jet')
            ax[1].set_title(f"Grad-CAM\nPredicted: {class_names[pred_class]}")
            ax[1].axis('off')

            ax[2].imshow(original_img_gray, cmap='gray', alpha=0.7)
            ax[2].imshow(cam_resized, cmap='jet', alpha=0.5)
            ax[2].set_title("Grad-CAM Overlay")
            ax[2].axis('off')

            plt.tight_layout()

            # Save
            actual_class_name = class_names[actual_class]
            pred_class_name = class_names[pred_class]
            filename = f"gradcam_batch{batch_idx}_img{i}_pred{pred_class_name}_actual{actual_class_name}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
            plt.show()
            plt.close()

            sample_count += 1

    print(f"Generated {sample_count} Grad-CAM visualizations in '{save_dir}' directory")

def generate_gradcam_for_trained_model():
    """Generate Grad-CAM visualizations using the trained MAE + MLP model"""

    # Define class names
    class_names = ['Non-referable DR', 'Referable DR']

    # File paths
    experiment_name = "DR_mlp_50000_optuna_bs16"
    finetuned_mlp_path = f'best_finetuned_{experiment_name}.pth'
    params_file = f'best_params_{experiment_name}.json'

    if not os.path.exists(finetuned_mlp_path):
        print(f"Trained MLP not found at {finetuned_mlp_path}")
        return

    # Load the best hyperparameters
    if os.path.exists(params_file):
        import json
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        best_hidden_dim = best_params['hidden_dim']
        best_dropout = best_params['dropout_rate']
        print(f"Loaded best hyperparameters: hidden_dim={best_hidden_dim}, dropout={best_dropout}")
    else:
        # Try to infer from the saved model
        print("Hyperparameters file not found. Trying to infer from saved model...")
        checkpoint = torch.load(finetuned_mlp_path, map_location=device)

        # Check the first layer weight shape to infer hidden_dim
        first_layer_weight = checkpoint['model.0.weight'] 
        best_hidden_dim = first_layer_weight.shape[0]
        best_dropout = 0.5 

        print(f"Inferred hyperparameters: hidden_dim={best_hidden_dim}, dropout={best_dropout}")

    # Create MLP classifier with correct hyperparameters
    mlp_classifier = MLPClassifier(
        input_dim=1024,
        hidden_dim=best_hidden_dim,
        num_classes=2,
        dropout_rate=best_dropout
    ).to(device)

    # Load trained weights
    mlp_classifier.load_state_dict(torch.load(finetuned_mlp_path, map_location=device))
    mlp_classifier.eval()

    print("Loaded trained MLP classifier")

    combined_model = CombinedMAE_MLP(pretrained_model, mlp_classifier).to(device)

    print("Created combined MAE + MLP model")

    # Generate Grad-CAM on test data
    test_dataloader = dataloaders['test']

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gradcam_dir = f'gradcam_maps_{experiment_name}_{timestamp}'

    print(f"Generating Grad-CAM visualizations...")
    generate_gradcam_mae_visualization(combined_model, test_dataloader, class_names, save_dir=gradcam_dir)

    print(f"Grad-CAM visualizations saved in '{gradcam_dir}' directory")

# ---------------------------- #
#       MLP DATASET
# ---------------------------- #
class FeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        # Filter for actual files ending with .pt
        self.feature_files = sorted([
            f for f in os.listdir(feature_dir)
            if os.path.isfile(os.path.join(feature_dir, f)) and f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.feature_dir, self.feature_files[idx]))
        features = data.get('features')
        label = data.get('label')
        return features, label

# ---------------------------- #
#         MLP CLASSIFIER
# ---------------------------- #
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim // 2, num_classes)
    )

    def forward(self, x):
        return self.model(x)

class CombinedMAE_MLP(nn.Module):
    """Combined MAE encoder + MLP classifier for end-to-end inference and saliency"""
    def __init__(self, mae_model, mlp_model):
        super().__init__()
        self.mae_encoder = mae_model
        self.mlp_classifier = mlp_model

    def forward(self, x):
        # Extract features using MAE encoder
        latent = self.mae_encoder.forward_encoder(x)
        cls_features = latent[:, 0, :]  

        # Classify using MLP
        logits = self.mlp_classifier(cls_features)
        return logits

# ---------------------------- #
#       OPTUNA TRAIN LOOP
# ---------------------------- #
def main_optuna():
    # ------------------- #
    #   Global Settings
    # ------------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = "DR_mlp_50000_optuna_bs16"
    finetuned_mlp = f'best_finetuned_{experiment_name}.pth'
    confusion_matrix_file = f'confusion_matrix_finetuned_{experiment_name}.png'

    # Paths to your feature directories
    train_feature_dir = os.path.join(feature_save_dir, 'train')
    val_feature_dir   = os.path.join(feature_save_dir, 'val')
    test_feature_dir  = os.path.join(feature_save_dir, 'test')

    # ------------------- #
    #   Data Loaders
    # ------------------- #
    train_dataset = FeatureDataset(train_feature_dir)
    val_dataset   = FeatureDataset(val_feature_dir)
    test_dataset  = FeatureDataset(test_feature_dir)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    # ------------------- #
    #  Objective Function
    # ------------------- #
    def objective(trial):

        lr           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        hidden_dim   = trial.suggest_int("hidden_dim", 64, 512, step=64)  
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.8)     
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) 

        input_dim   = 1024   
        num_classes = 2     

        classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        ).to(device)


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(  
            classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 7  # Stop if no improvement for 7 epochs

        max_epochs = 20  
        for epoch in range(max_epochs):
            classifier.train()
            running_loss = 0.0
            correct = 0

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = classifier(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * features.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / len(train_loader.dataset)

            # Validation
            classifier.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    logits = classifier(features)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * features.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = classifier.state_dict().copy()  # Save best model
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            print(f"Trial {trial.number}, Epoch {epoch+1}/{max_epochs}, "
                  f"LR={lr:.5f}, Hidden={hidden_dim}, Dropout={dropout_rate:.2f}, WeightDecay={weight_decay:.6f},"
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


            scheduler.step(val_loss)

            # (Optional) Pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    # ------------------- #
    # Run the Study
    # ------------------- #
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    df = study.trials_dataframe()
    csv_path = f'optuna_trials_{experiment_name}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved Optuna trials to {csv_path}")

    # Print best hyperparameters
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # ------------------- #
    # Save best hyperparameters
    # ------------------- #
    best_params = {
        'lr': trial.params["lr"],
        'hidden_dim': trial.params["hidden_dim"],
        'dropout_rate': trial.params["dropout_rate"],
        'weight_decay': trial.params["weight_decay"]
    }

    import json
    params_file = f'best_params_{experiment_name}.json'
    with open(params_file, 'w') as f:
        json.dump(best_params, f)
    print(f"Saved best parameters to {params_file}")

    # ------------------- #
    # Retrain with best hyperparams
    # ------------------- #
    best_lr           = trial.params["lr"]
    best_hidden_dim   = trial.params["hidden_dim"]
    best_dropout      = trial.params["dropout_rate"]
    best_weight_decay = trial.params["weight_decay"]

    final_classifier = MLPClassifier(
        input_dim=1024,
        hidden_dim=best_hidden_dim,
        num_classes=2,
        dropout_rate=best_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        final_classifier.parameters(),
        lr=best_lr,
        weight_decay=best_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    epochs = 50

    # Track the best model by validation loss
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(epochs):
        # ---- Train ----
        final_classifier.train()
        running_loss = 0.0
        correct = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = final_classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_classifier.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        # ---- Validate ----
        final_classifier.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).long()
                logits = final_classifier(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"[Final Retrain] Epoch {epoch+1}/{epochs}, "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        scheduler.step(val_loss)

        # Keep the best weights by lowest val_loss
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in final_classifier.state_dict().items()}
            best_epoch = epoch + 1

    # Save the BEST weights
    torch.save(best_state, finetuned_mlp)
    print(f"Saved best MLP weights to {finetuned_mlp} (epoch {best_epoch}, val_loss={best_val:.4f})")

    # Save a metadata checkpoint
    ckpt_path = finetuned_mlp.replace(".pth", "_ckpt.pth")
    checkpoint = {
        "model_type": "MLPClassifier",
        "state_dict": best_state,
        "input_dim": 1024,
        "num_classes": 2,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "hyperparams": {
            "lr": best_lr,
            "hidden_dim": best_hidden_dim,
            "dropout_rate": best_dropout,
            "weight_decay": best_weight_decay
        },
        "label_map": {0: "non-referable", 1: "referable"},
        "pretrained_encoder_ckpt_path": pretrained_weights,
        "feature_dir_version": os.path.basename(feature_save_dir),
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved full checkpoint to {ckpt_path}")


    # ------------------- #
    #        Test
    # ------------------- #
    final_classifier.load_state_dict(best_state)
    final_classifier.eval()
    test_loss = 0.0
    test_correct = 0
    test_all_preds = []
    test_all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = final_classifier(features)
            loss = criterion(logits, labels)
            test_loss += loss.item() * features.size(0)

            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()

            test_all_preds.extend(preds.cpu().numpy())
            test_all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    print(f"\n>>> Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print("\nTesting different thresholds:")
    final_classifier.eval()
    with torch.no_grad():
        for threshold in [0.3, 0.4, 0.5, 0.6]:
            threshold_preds = []

            for features, labels in test_loader:
                features = features.to(device)
                logits = final_classifier(features)
                probs = torch.softmax(logits, dim=1)
                referable_probs = probs[:, 1]  
                preds = (referable_probs > threshold).long()
                threshold_preds.extend(preds.cpu().numpy())

            # Calculate metrics with this threshold
            correct = sum(p == l for p, l in zip(threshold_preds, test_all_labels))
            accuracy = correct / len(test_all_labels)

            # Calculate referable recall
            referable_correct = sum(p == 1 and l == 1 for p, l in zip(threshold_preds, test_all_labels))
            referable_total = sum(l == 1 for l in test_all_labels)
            sensitivity = referable_correct / referable_total if referable_total > 0 else 0

            print(f"Threshold: {threshold}, Accuracy: {accuracy:.4f}, Referable Recall: {sensitivity:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(test_all_labels, test_all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-referable', 'Referable'], 
                yticklabels=['Non-referable', 'Referable'])
    plt.title('Confusion Matrix - Optuna Best')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(confusion_matrix_file)
    plt.close()