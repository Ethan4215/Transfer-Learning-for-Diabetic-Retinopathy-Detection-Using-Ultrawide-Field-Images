import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch
from torch.utils.data import DataLoader, Dataset
import os
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

plt.style.use('default')
sns.set_palette("husl")

# This is for the foundation model

def load_features_and_labels(feature_dir):
    """
    Load all features and labels from your feature directory
    Assumes your FeatureDataset structure
    """
    features = []
    labels = []

    # Iterate through all .pt files in the directory
    for filename in os.listdir(feature_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(feature_dir, filename)
            data = torch.load(filepath, map_location='cpu')

            # Extract features and labels based on your data structure
            if isinstance(data, dict):
                features.append(data['features'])
                labels.append(data['label'])
            else:
                # If it's just the features, you'll need to infer labels from filename
                features.append(data)
                if 'class_0' in filename or 'negative' in filename:
                    labels.append(0)
                else:
                    labels.append(1)

    # Convert to numpy arrays
    features = np.vstack([f.numpy() if isinstance(f, torch.Tensor) else f for f in features])
    labels = np.array(labels)

    return features, labels

def perform_dimensionality_reduction(features, labels, method='both', random_state=42):
    """
    Perform t-SNE and/or UMAP on features
    """
    results = {}

    if method in ['tsne', 'both']:
        print("Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=30,  
            n_iter=1000,
            verbose=1
        )
        tsne_result = tsne.fit_transform(features)
        results['tsne'] = tsne_result

    if method in ['umap', 'both']:
        print("Running UMAP...")
        umap_reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=15,  
            min_dist=0.1
        )
        umap_result = umap_reducer.fit_transform(features)
        results['umap'] = umap_result

    return results

def plot_embeddings(embeddings_dict, labels, model_name="Model", save_path=None):
    """
    Plot t-SNE and UMAP results
    """
    n_plots = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    colors = ['red', 'blue']  
    class_names = ['Non-referable', 'Referable']  

    for idx, (method, embedding) in enumerate(embeddings_dict.items()):
        ax = axes[idx]

        # Create scatter plot
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=colors[class_idx],
                label=class_names[class_idx],
                alpha=0.7,
                s=50
            )

        ax.set_title(f'{method.upper()} - {model_name}')
        ax.set_xlabel(f'{method.upper()}-1')
        ax.set_ylabel(f'{method.upper()}-2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def analyze_separation(embeddings, labels):
    """
    Quantitative analysis of class separation
    """
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    # Silhouette score
    sil_score = silhouette_score(embeddings, labels)

    # Calculate inter-class vs intra-class distances
    class_0_points = embeddings[labels == 0]
    class_1_points = embeddings[labels == 1]

    # Intra-class distances (within same class)
    if len(class_0_points) > 1:
        intra_0 = cdist(class_0_points, class_0_points).mean()
    else:
        intra_0 = 0

    if len(class_1_points) > 1:
        intra_1 = cdist(class_1_points, class_1_points).mean()
    else:
        intra_1 = 0

    # Inter-class distances (between different classes)
    if len(class_0_points) > 0 and len(class_1_points) > 0:
        inter_dist = cdist(class_0_points, class_1_points).mean()
    else:
        inter_dist = 0

    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Average Intra-class Distance (Non-referable): {intra_0:.4f}")
    print(f"Average Intra-class Distance (Referable): {intra_1:.4f}")
    print(f"Average Inter-class Distance: {inter_dist:.4f}")

    # Higher inter/intra ratio = better separation
    avg_intra = (intra_0 + intra_1) / 2
    if avg_intra > 0:
        separation_ratio = inter_dist / avg_intra
        print(f"Separation Ratio (higher = better): {separation_ratio:.4f}")

    return {
        'silhouette_score': sil_score,
        'intra_class_0': intra_0,
        'intra_class_1': intra_1,
        'inter_class': inter_dist
    }

# Simple function to analyze just your MAE+ViT features
def analyze_mae_vit_features():
    """
    Simple analysis of just your MAE+ViT features
    """
    train_feature_dir = '/content/drive/My Drive/ultra-widefield_images_soft/featuresv6/train'  
    val_feature_dir = '/content/drive/My Drive/ultra-widefield_images_soft/featuresv6/val'     

    class FeatureDataset(Dataset):
        def __init__(self, feature_dir):
            self.feature_dir = feature_dir
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

    train_dataset = FeatureDataset(train_feature_dir)
    val_dataset = FeatureDataset(val_feature_dir)

    print("Loading MAE+ViT features...")

    # Extract all features and labels
    train_features = []
    train_labels = []
    for features, label in train_dataset:
        train_features.append(features.numpy())
        train_labels.append(label)

    val_features = []
    val_labels = []
    for features, label in val_dataset:
        val_features.append(features.numpy())
        val_labels.append(label)

    # Combine train and validation
    all_features = np.vstack(train_features + val_features)
    all_labels = np.array(train_labels + val_labels)

    print(f"Total samples: {len(all_labels)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Class distribution: {np.bincount(all_labels)}")

    # Run t-SNE and UMAP
    print("\nRunning dimensionality reduction...")
    embeddings = perform_dimensionality_reduction(all_features, all_labels)

    # Create plots
    plot_embeddings(embeddings, all_labels, "MAE+ViT", "mae_vit_embeddings.png")

    # Print separation analysis
    print("\nSeparation Analysis:")
    for method, embedding in embeddings.items():
        print(f"\n--- {method.upper()} ---")
        analyze_separation(embedding, all_labels)

    return embeddings, all_labels

# This is for the ResNet-18 Model

def extract_resnet_features(model, dataloader, device, layer_name='avgpool'):
    """
    Extract features from ResNet before the final classification layer
    """
    model.eval()
    features = []
    labels = []

    # Hook to extract features from the specified layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register hook (avgpool is the layer before fc in ResNet)
    if layer_name == 'avgpool':
        model.avgpool.register_forward_hook(get_activation('avgpool'))
    elif layer_name == 'features':
        # For features before avgpool (more detailed)
        model.layer4.register_forward_hook(get_activation('features'))

    print(f"Extracting features from {layer_name} layer...")

    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get the features from the hook
            if layer_name == 'avgpool':
                batch_features = activation['avgpool'].squeeze()  # Remove spatial dimensions
            elif layer_name == 'features':
                batch_features = activation['features']
                # Global average pooling if using features from layer4
                batch_features = torch.mean(batch_features, dim=(2, 3))

            # Handle single sample case
            if batch_features.dim() == 1:
                batch_features = batch_features.unsqueeze(0)

            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())

    # Combine all features
    all_features = np.vstack(features)
    all_labels = np.array(labels)

    print(f"Extracted features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")

    return all_features, all_labels

def perform_dimensionality_reduction(features, labels, method='both', random_state=42):
    """
    Perform t-SNE and/or UMAP on features
    """
    results = {}

    if method in ['tsne', 'both']:
        print("Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, len(features)//4),  # Adjust perplexity for small datasets
            n_iter=1000,
            verbose=1
        )
        tsne_result = tsne.fit_transform(features)
        results['tsne'] = tsne_result

    if method in ['umap', 'both']:
        print("Running UMAP...")
        umap_reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=min(15, len(features)//3),  # Adjust for small datasets
            min_dist=0.1
        )
        umap_result = umap_reducer.fit_transform(features)
        results['umap'] = umap_result

    return results

def plot_embeddings(embeddings_dict, labels, model_name="Model", save_path=None):
    """
    Plot t-SNE and UMAP results
    """
    n_plots = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    colors = ['red', 'blue']
    class_names = ['Non-referable', 'Referable']  # Based on your dataset

    for idx, (method, embedding) in enumerate(embeddings_dict.items()):
        ax = axes[idx]

        # Create scatter plot
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=colors[class_idx],
                label=class_names[class_idx],
                alpha=0.7,
                s=50
            )

        ax.set_title(f'{method.upper()} - {model_name}')
        ax.set_xlabel(f'{method.upper()}-1')
        ax.set_ylabel(f'{method.upper()}-2')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def analyze_separation(embeddings, labels):
    """
    Quantitative analysis of class separation
    """
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    # Silhouette score (higher is better, range [-1, 1])
    sil_score = silhouette_score(embeddings, labels)

    # Calculate inter-class vs intra-class distances
    class_0_points = embeddings[labels == 0]
    class_1_points = embeddings[labels == 1]

    # Intra-class distances (within same class)
    if len(class_0_points) > 1:
        intra_0 = cdist(class_0_points, class_0_points).mean()
    else:
        intra_0 = 0

    if len(class_1_points) > 1:
        intra_1 = cdist(class_1_points, class_1_points).mean()
    else:
        intra_1 = 0

    # Inter-class distances (between different classes)
    if len(class_0_points) > 0 and len(class_1_points) > 0:
        inter_dist = cdist(class_0_points, class_1_points).mean()
    else:
        inter_dist = 0

    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Average Intra-class Distance (Non-referable): {intra_0:.4f}")
    print(f"Average Intra-class Distance (Referable): {intra_1:.4f}")
    print(f"Average Inter-class Distance: {inter_dist:.4f}")

    # Higher inter/intra ratio = better separation
    avg_intra = (intra_0 + intra_1) / 2
    if avg_intra > 0:
        separation_ratio = inter_dist / avg_intra
        print(f"Separation Ratio (higher = better): {separation_ratio:.4f}")

    return {
        'silhouette_score': sil_score,
        'intra_class_0': intra_0,
        'intra_class_1': intra_1,
        'inter_class': inter_dist
    }

def analyze_resnet_features(model, dataloaders, device):
    """
    Main function to analyze ResNet features
    """
    print("=== ResNet Feature Analysis ===")

    # Combine train and val data for analysis
    all_features = []
    all_labels = []

    for phase in ['train', 'val']:
        print(f"\nExtracting {phase} features...")
        features, labels = extract_resnet_features(
            model, dataloaders[phase], device, layer_name='avgpool'
        )
        all_features.append(features)
        all_labels.append(labels)

    # Combine all data
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)

    print(f"\nCombined features shape: {combined_features.shape}")
    print(f"Class distribution: {np.bincount(combined_labels)}")

    # Run dimensionality reduction
    print("\nRunning dimensionality reduction...")
    embeddings = perform_dimensionality_reduction(combined_features, combined_labels)

    # Create plots
    plot_embeddings(embeddings, combined_labels, "ResNet-18", "resnet_embeddings.png")

    # Analyze separation
    print("\nResNet-18 Separation Analysis:")
    for method, embedding in embeddings.items():
        print(f"\n--- {method.upper()} ---")
        analyze_separation(embedding, combined_labels)

    return embeddings, combined_labels

# Usage example:
def run_resnet_analysis():
    """
    Run this after you've trained your ResNet model
    """
    # Important note: Run this after the ResNet model has been trained

    # Make sure model is in eval mode
    model.eval()

    # Run the analysis
    embeddings, labels = analyze_resnet_features(model, dataloaders, device)

    print("\n=== Analysis Complete ===")
    print("Check the saved plot: resnet_embeddings.png")

    return embeddings, labels