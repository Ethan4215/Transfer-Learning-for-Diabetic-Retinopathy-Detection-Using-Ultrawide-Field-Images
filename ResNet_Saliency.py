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
import torch.nn.functional as F

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
def train_model(model, dataloaders, criterion_train, criterion_val, optimizer, num_epochs=20, patience=4):
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
                if epoch_acc > best_acc:  
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

def generate_saliency_map(model, dataloader, class_names, save_dir='saliency_maps', num_samples=10):
    """
    Generate saliency maps showing which parts of images the model focuses on.
    """
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)
    
    samples_processed = 0
    print(f"Generating saliency maps for {num_samples} samples...")
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        for i in range(inputs.size(0)):
            if samples_processed >= num_samples:
                break
                
            # Get single image
            image = inputs[i:i+1]
            true_label = labels[i].item()
            
            # Compute saliency
            image.requires_grad_(True)
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = F.softmax(output, dim=1)[0, predicted_class].item()
            
            # Backward pass to get gradients
            model.zero_grad()
            output[0, predicted_class].backward()
            
            # Create saliency map from gradients
            saliency = torch.max(torch.abs(image.grad.data), dim=1)[0] 
            saliency = saliency.squeeze(0)  
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            
            # Prepare image for display (denormalize)
            img_display = image.squeeze(0).detach().cpu().numpy()  
            img_display = (img_display * 0.5) + 0.5
            img_display = np.clip(img_display.transpose(1, 2, 0), 0, 1)  
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original image
            axes[0].imshow(img_display)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Saliency map
            axes[1].imshow(saliency.cpu().numpy(), cmap='hot')
            axes[1].set_title('Saliency Map')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_display)
            axes[2].imshow(saliency.cpu().numpy(), cmap='jet', alpha=0.4)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            # Add prediction info
            correct = "✓" if true_label == predicted_class else "✗"
            color = 'green' if true_label == predicted_class else 'red'
            
            plt.suptitle(f'True: {class_names[true_label]} | Pred: {class_names[predicted_class]} | '
                        f'Conf: {confidence:.3f} {correct}', 
                        color=color, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{samples_processed:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            samples_processed += 1
    
    print(f"Saved {samples_processed} saliency maps to {save_dir}")

if __name__ == "__main__":
    # 1. Train model on train and validation sets
    trained_model = train_model(model, dataloaders, criterion_train, criterion_val, optimizer, num_epochs=20, patience=4)

    # 2. Test model and compute confusion matrix, AUROC, AUPR and Cohen's Kappa
    labels, preds, auroc, aupr, kappa = test_model(trained_model, dataloaders['test'], criterion_test)
    class_names = datasets_dict['train'].classes
    save_path = "resnet_confusion_matrix.png"
    plot_confusion_matrix(labels, preds, class_names, save_path=save_path)
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Test AUROC: {auroc}")
    print(f"Test AUPR: {aupr}")

    generate_saliency_map(model, dataloaders['test'], class_names, save_dir='saliency_both_classes')