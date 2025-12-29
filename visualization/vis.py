import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments.multiclass_verification import get_mnist_data, NUM_POSITIONS, positions
from utils.utils import TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT

pos_names = {TOP_LEFT: "Top-Left", TOP_RIGHT: "Top-Right", 
             BOTTOM_LEFT: "Bottom-Left", BOTTOM_RIGHT: "Bottom-Right"}

os.makedirs("visualization_outputs", exist_ok=True)


def visualize_grid(num_samples=100, use_patches=True, grid_shape=(4, 4), save_path="visualization_outputs/grid_examples.png"):
    train_dataset, _ = get_mnist_data(num_samples=num_samples, validation_size=50, use_patches=use_patches)
    
    images, labels = [], []
    for img, label in train_dataset:
        images.append(img.numpy())
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    num_images = min(len(images), grid_shape[0] * grid_shape[1])
    indices = np.random.choice(len(images), num_images, replace=False)
    
    fig, axes = plt.subplots(grid_shape[0], grid_shape[1], figsize=(grid_shape[1] * 2, grid_shape[0] * 2))
    axes = axes.flatten()
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        img = images[ax_idx].reshape(28, 28)
        syn_label = labels[ax_idx]
        
        true_label = syn_label // NUM_POSITIONS
        pos_idx = syn_label % NUM_POSITIONS
        patch_pos = positions[pos_idx]
        
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        
        title = f"True: {true_label}\nSyn: {syn_label}\n{pos_names[patch_pos]}"
        ax.set_title(title, fontsize=9)
    
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid to {save_path}")


def visualize_single_by_synthetic_class(synthetic_class, num_samples=100, use_patches=True, 
                                       save_path="visualization_outputs/single_example.png"):
    train_dataset, _ = get_mnist_data(num_samples=num_samples, validation_size=50, use_patches=use_patches)
    
    images, labels = [], []
    for img, label in train_dataset:
        images.append(img.numpy())
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    matching_indices = np.where(labels == synthetic_class)[0]
    if len(matching_indices) == 0:
        print(f"No samples found for synthetic class {synthetic_class}")
        return
    
    idx = np.random.choice(matching_indices)
    img = images[idx].reshape(28, 28)
    syn_label = labels[idx]
    
    true_label = syn_label // NUM_POSITIONS
    pos_idx = syn_label % NUM_POSITIONS
    patch_pos = positions[pos_idx]
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    
    title = f"True Class: {true_label}\nSynthetic Class: {syn_label}\nPatch: {pos_names[patch_pos]}"
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved single example to {save_path}")


if __name__ == "__main__":
    visualize_grid(num_samples=100, use_patches=True, grid_shape=(4, 4))
    visualize_single_by_synthetic_class(synthetic_class=0, num_samples=100, use_patches=True)
