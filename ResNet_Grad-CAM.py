import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import random
from sklearn.metrics import classification_report
from torch.utils.data import WeightedRandomSampler

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Directories --------- #
data_dir = '/content/drive/My Drive/ultra-widefield_images_soft'

# ---------------------------- #
#        Model Setup
# ---------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2 # binary classification = referable vs. non-referable

# Load a ResNet-18 model pretrained on ImageNet
model = resnet18(pretrained=True)


# Modify the final layer to match the number of classes (e.g. 2)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to the desired device
model = model.to(device)

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

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()


# ---------------------------- #
#       Training Setup
# ---------------------------- #
criterion_train = nn.CrossEntropyLoss()
criterion_val = nn.CrossEntropyLoss()
criterion_test = nn.CrossEntropyLoss()
# ---------------------------- #
#         Training Loop
# ---------------------------- #
def train_model(model, dataloaders, criterion_train, criterion_val, optimizer, num_epochs=20, patience=8):
    best_model_weights = model.state_dict()
    best_acc = 0.0
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            criterion = criterion_train if phase == 'train' else criterion_val
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val':
                if epoch_acc > best_acc:  # use accuracy as early stopping signal
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_weights = model.state_dict()
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1


        # Early stopping condition
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best Validation Loss: {best_loss:.4f}, Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_weights)
    return model

# ---------------------------- #
#       Testing & Metrics
# ---------------------------- #
def test_model(model, dataloader, criterion):
    model.eval()
    running_corrects = 0
    all_labels = []
    all_preds = []
    all_probs = []
    class_names = ['non_referable', 'referable']

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    test_acc = running_corrects.double() / len(dataloader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    # ============ FIXED AUROC/AUPR CALCULATION ============
    try:
        # For binary classification, use probabilities of positive class (class 1) directly
        # Don't use label_binarize for binary classification
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
        aupr = average_precision_score(all_labels, all_probs[:, 1])

        print(f"AUROC: {auroc:.4f}")
        print(f"AUPR: {aupr:.4f}")
    except Exception as e:
        auroc, aupr = None, None
        print(f"Could not compute AUROC/AUPR: {e}")

    # Compute Cohen's Kappa
    kappa_score = cohen_kappa_score(all_labels, all_preds)
    print(f"Cohen's Kappa: {kappa_score:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return all_labels, all_preds, auroc, aupr, kappa_score

def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, format='png')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

import cv2

def get_gradcam(model, input_tensor, target_class_idx, target_layer_name='layer4'):

    model.eval()

    # Storage for gradients and activations
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    target_layer = dict(model.named_modules())[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    try:
        # Forward pass
        output = model(input_tensor)

        # Backward pass
        model.zero_grad()
        class_score = output[:, target_class_idx].sum()
        class_score.backward(retain_graph=True)

        # Get gradients and activations
        grads = gradients[0].cpu().data.numpy()  
        acts = activations[0].cpu().data.numpy() 

        # Generate CAM
        batch_size, num_channels, h, w = grads.shape

        # Global average pooling of gradients
        weights = np.mean(grads, axis=(2, 3)) 

        # Generate CAM for each image in batch
        cams = []
        for i in range(batch_size):
            cam = np.zeros((h, w), dtype=np.float32)
            for j in range(num_channels):
                cam += weights[i, j] * acts[i, j, :, :]

            # Apply ReLU 
            cam = np.maximum(cam, 0)

            if cam.max() > 0:
                cam = cam / cam.max()

            cams.append(cam)

        return np.array(cams)

    finally:
        # Clean up hooks
        forward_handle.remove()
        backward_handle.remove()

def generate_gradcam_visualization(model, dataloader, class_names, save_dir='gradcam_maps'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Collect all samples from the dataset
    all_samples = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        for i in range(inputs.size(0)):
            all_samples.append((inputs[i], labels[i], batch_idx, i))

    # Randomly select 5 samples
    max_samples = 5
    random_indices = random.sample(range(len(all_samples)), min(max_samples, len(all_samples)))
    selected_samples = [all_samples[idx] for idx in random_indices]

    print(f"Selected {len(selected_samples)} random samples from {len(all_samples)} total samples")

    # Process each selected sample
    for sample_idx, (input_tensor, label_tensor, orig_batch_idx, orig_img_idx) in enumerate(selected_samples):
        # Add batch dimension and move to device
        single_input = input_tensor.unsqueeze(0).to(device)
        single_input.requires_grad = True
        single_label = label_tensor.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            outputs = model(single_input)
            _, predicted = torch.max(outputs, 1)

        pred_class = predicted[0].item()

        # Generate Grad-CAM
        cam = get_gradcam(model, single_input, pred_class)
        cam = cam[0]  

        input_height, input_width = single_input.shape[2], single_input.shape[3]
        cam_resized = cv2.resize(cam, (input_width, input_height))

        original_img = single_input[0].detach().cpu()
        original_img = (original_img * 0.5) + 0.5  
        original_img = torch.clamp(original_img, 0, 1)
        original_img = original_img.permute(1, 2, 0).numpy()

        # Create visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        ax[0].imshow(original_img)
        ax[0].set_title(f"Original\nActual: {class_names[single_label[0]]}")
        ax[0].axis('off')

        # Grad-CAM heatmap
        ax[1].imshow(cam_resized, cmap='jet')
        ax[1].set_title(f"Grad-CAM\nPredicted: {class_names[pred_class]}")
        ax[1].axis('off')

        # Overlay
        ax[2].imshow(original_img, alpha=0.7)
        ax[2].imshow(cam_resized, cmap='jet', alpha=0.5)
        ax[2].set_title("Grad-CAM Overlay")
        ax[2].axis('off')

        plt.tight_layout()

        # Save
        actual_class = class_names[single_label[0]]
        pred_class_name = class_names[pred_class]
        filename = f"gradcam_random{sample_idx}_pred{pred_class_name}_actual{actual_class}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    print(f"Generated {len(selected_samples)} Grad-CAM visualizations in '{save_dir}' directory")

def generate_gradcam_for_both_classes(model, dataloader, class_names, save_dir='gradcam_both_classes'):

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Collect all samples from the dataset
    all_samples = []
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        for i in range(inputs.size(0)):
            all_samples.append((inputs[i], labels[i], batch_idx, i))

    # Randomly select 5 samples
    max_samples = 5
    random_indices = random.sample(range(len(all_samples)), min(max_samples, len(all_samples)))
    selected_samples = [all_samples[idx] for idx in random_indices]

    print(f"Selected {len(selected_samples)} random samples from {len(all_samples)} total samples")

    # Process each selected sample
    for sample_idx, (input_tensor, label_tensor, orig_batch_idx, orig_img_idx) in enumerate(selected_samples):
        single_input = input_tensor.unsqueeze(0).to(device)
        single_input.requires_grad = True
        single_label = label_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(single_input)
            _, predicted = torch.max(outputs, 1)

        pred_class = predicted[0].item()
        actual_class = single_label[0].item()

        cam_class0 = get_gradcam(model, single_input, 0)[0]  # Class 0
        cam_class1 = get_gradcam(model, single_input, 1)[0]  # Class 1

        input_height, input_width = single_input.shape[2], single_input.shape[3]
        cam0_resized = cv2.resize(cam_class0, (input_width, input_height))
        cam1_resized = cv2.resize(cam_class1, (input_width, input_height))

        original_img = single_input[0].detach().cpu()
        original_img = (original_img * 0.5) + 0.5
        original_img = torch.clamp(original_img, 0, 1)
        original_img = original_img.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))

        # Original
        ax[0].imshow(original_img)
        ax[0].set_title(f"Original\nPred: {class_names[pred_class]}\nActual: {class_names[actual_class]}")
        ax[0].axis('off')

        # Grad-CAM for class 0
        ax[1].imshow(original_img, alpha=0.7)
        ax[1].imshow(cam0_resized, cmap='jet', alpha=0.5)
        ax[1].set_title(f"Grad-CAM for\n{class_names[0]}")
        ax[1].axis('off')

        # Grad-CAM for class 1
        ax[2].imshow(original_img, alpha=0.7)
        ax[2].imshow(cam1_resized, cmap='jet', alpha=0.5)
        ax[2].set_title(f"Grad-CAM for\n{class_names[1]}")
        ax[2].axis('off')

        # Difference 
        cam_diff = cam1_resized - cam0_resized
        ax[3].imshow(original_img, alpha=0.7)
        im = ax[3].imshow(cam_diff, cmap='RdBu_r', alpha=0.6, vmin=-1, vmax=1)
        ax[3].set_title(f"Class Difference\n({class_names[1]} - {class_names[0]})")
        ax[3].axis('off')

        plt.tight_layout()

        # Save
        status = "CORRECT" if pred_class == actual_class else "WRONG"
        filename = f"gradcam_random{sample_idx}_pred{class_names[pred_class]}_actual{class_names[actual_class]}_{status}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

    print(f"Generated {len(selected_samples)} dual-class Grad-CAM visualizations in '{save_dir}' directory")
