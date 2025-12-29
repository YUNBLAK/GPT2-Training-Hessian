import os
# Limit thread usage to prevent overheating on laptop
# Set these BEFORE importing/initializing torch/numpy
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads (used by many libraries)
os.environ["MKL_NUM_THREADS"] = "4"  # Intel MKL threads (if using Intel MKL)
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS threads (alternative BLAS)
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr threads (used by some numpy operations)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from torchvision import datasets, transforms
from contextlib import nullcontext
import argparse
from model import MLPSimple
import utils.hessian_spectrum as hessian_mod
from utils.utils import TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, add_one_patch_mnist


torch.set_float32_matmul_precision('high')

"""
Train a simple MLP neural network on MNIST dataset using PyTorch.
Extract the Hessian Matrix and plot normalized (divided by the 10th largest eigenvalue) eigenvalues for each layer in the network.
"""

NUM_TRUE_CLASSES = 10

# Define patch positions for 40 classes
positions = [TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]
NUM_POSITIONS = len(positions)
NUM_SYNTHETIC_CLASSES = NUM_TRUE_CLASSES * NUM_POSITIONS  # 10 * 4 = 40


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def get_mnist_data(num_samples=None, validation_size=50, use_patches=False):
    """
    Fetch MNIST datasets based on given parameters.

    Args:
        num_samples: Limit number of samples (None = all). Remark: this applies to original samples,
                    so with patches, you'll get num_samples * 4 images.
        use_patches: If True, generate synthetic classes by applying patches at 4 positions.
                    Each original image becomes 4 images with different patch positions.
    Returns:
        train_dataset: torch.utils.data.Dataset
        validation_dataset: torch.utils.data.Dataset
    """

    data_root = './data' if os.path.exists('./data') else '../data'
    main_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    
    ' Get validation subset from test set (D_val) '
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transforms.ToTensor())
    unique_labels_val = torch.unique(torch.tensor(test_dataset.targets))
    validation_indices = []

    for label in unique_labels_val:
        segmented_indicies = torch.where(test_dataset.targets == label)[0]
        selected_indices = segmented_indicies[torch.randperm(len(segmented_indicies))[:validation_size]]
        validation_indices.extend(selected_indices.tolist())
    
    validation_subset = torch.utils.data.Subset(test_dataset, validation_indices)

    ' Get training subset from main dataset (D_tr) '
    unique_labels_tr = torch.unique(torch.tensor(main_dataset.targets))
    train_indices = []

    for label in unique_labels_tr:
        segmented_indicies = torch.where(main_dataset.targets == label)[0]
        if num_samples is not None:
            selected_indices = segmented_indicies[torch.randperm(len(segmented_indicies))[:num_samples]]
            train_indices.extend(selected_indices.tolist())
        else:
            train_indices.extend(segmented_indicies.tolist())

    train_subset = torch.utils.data.Subset(main_dataset, train_indices)

    # Process datasets
    images, labels, images_val, labels_val = [], [], [], []
    
    ' Process validation subset '
    for img, label in validation_subset:
        if use_patches:
            img_2d = img.squeeze(0)
            for pos_idx, position in enumerate(positions):
                patched_img = add_one_patch_mnist(img_2d, position, patch_size=5)
                images_val.append(patched_img.numpy().flatten())
                labels_val.append(label * NUM_POSITIONS + pos_idx)
        else:
            images_val.append(img.numpy().flatten())
            labels_val.append(label)

    ' Process training subset '
    for img, label in train_subset:
        if use_patches:
            img_2d = img.squeeze(0)
            for pos_idx, position in enumerate(positions):
                patched_img = add_one_patch_mnist(img_2d, position, patch_size=5)
                images.append(patched_img.numpy().flatten())
                labels.append(label * NUM_POSITIONS + pos_idx)
        else:
            images.append(img.numpy().flatten())
            labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    images_val = np.array(images_val, dtype=np.float32)
    labels_val = np.array(labels_val, dtype=np.int64)
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(images_val), torch.from_numpy(labels_val))
    return train_dataset, validation_dataset


class MNISTHessian(hessian_mod.Hessian):
    """ Extended Hessian class for MNIST that handles image/label batching. """
    def __init__(self, model, train_images, train_labels, **kwargs):
        # Store images and labels separately
        self.train_images = train_images
        self.train_labels = train_labels
        self.num_samples = len(train_images)
        
        # Create a dummy train_data array for compatibility
        # We'll override get_batch anyway
        dummy_data = np.zeros(len(train_images) * 784, dtype=np.float32)
        
        # Initialize parent with dummy data (we override get_batch)
        super().__init__(
            model=model,
            train_data=dummy_data,
            block_size=784, # 28*28 mnist dimsonsions
            **kwargs
        )
        
        # Override num_batches calculation
        self.num_batches = self.num_samples // self.batch_size
        if self.num_batches == 0:
            self.num_batches = 1
        print(f'[MNIST] Total samples: {self.num_samples}, batches: {self.num_batches}')
    
    def get_batch(self, batch_idx):
        """
        Override to return MNIST images and labels.
        Returns:
            X: Images of shape (batch_size, 784)
            Y: Labels of shape (batch_size,)
        """
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, self.num_samples)
        
        # Get batch of images and labels
        batch_images = self.train_images[start_idx:end_idx]
        batch_labels = self.train_labels[start_idx:end_idx]
        
        # Convert to tensors
        X = torch.from_numpy(batch_images).float()
        Y = torch.from_numpy(batch_labels).long()
        
        # Move to device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        return X, Y


def get_args():
    parser = argparse.ArgumentParser(description="Hessian spectrum for MNIST MLP")
    parser.add_argument("--ckpt", type=str, default=None,
                       help="path to checkpoint .pt file (optional, will train if not provided)")
    parser.add_argument("--outdir", type=str, required=True,
                       help="tag or folder name used by Hessian(comment) to save plots")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="batch size used in Hessian computation")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="gradient accumulation steps used in Hessian class")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="number of MNIST samples to use (None = all 60k)")
    parser.add_argument("--m", type=int, default=100,
                       help="number of Lanczos basis vectors")
    parser.add_argument("--num_v", type=int, default=10,
                       help="number of random directions for Lanczos")
    parser.add_argument("--train_epochs", type=int, default=5,
                       help="number of epochs to train if no checkpoint provided")
    parser.add_argument("--train_lr", type=float, default=0.001,
                       help="learning rate for training")
    parser.add_argument("--use_patches", action="store_true", default=False,
                       help="Use synthetic patch-based classes. Creates 40 classes (10 true classes × 4 patch positions). Default: False (normal 10-class classification)")
    parser.add_argument("--validation_size", type=int, default=50,
                       help="number of validation samples") # make this proportional to num_samples
    return parser.parse_args()


def train_model(model, device, train_loader, epochs=5, lr=1e-4):
    """ Train the MLP model on MNIST. """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"[INFO] Training model for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(data, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        print(f'Epoch {epoch} completed. Avg Loss: {total_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}%')
    model.eval()
    return model


def load_or_train_model(device, args, use_patches=False):
    """ Load model from checkpoint or train a new one. """
    num_classes = NUM_SYNTHETIC_CLASSES if use_patches else 10
    
    if args.ckpt and os.path.exists(args.ckpt):
        ' Load model from checkpoint. '

        print(f"[INFO] Loading model from checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

        model = MLPSimple(layer_sizes=[8, num_classes])
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        # Get validation dataset for accuracy evaluation
        _, validation_dataset = get_mnist_data(num_samples=args.num_samples, validation_size=args.validation_size, use_patches=use_patches)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

        return model, None, validation_loader # Return None for trained_epochs since we don't know how many epochs it was trained for
    else:
        ' Train new model. '
        if use_patches:
            mode_str = f"{NUM_SYNTHETIC_CLASSES}-class synthetic MNIST (patches at {NUM_POSITIONS} positions)"
        else:
            mode_str = "10-class"
        print(f"[INFO] No checkpoint provided, training new model ({mode_str} classification)...")
        
        model = MLPSimple(layer_sizes=[8, num_classes])
        model.to(device)
        
        # Get training and validation datasets
        train_dataset, validation_dataset = get_mnist_data(num_samples=args.num_samples, validation_size=args.validation_size, use_patches=use_patches)        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = train_model(model, device, train_loader, epochs=args.train_epochs, lr=args.train_lr)
        
        ' Save checkpoint. '
        if use_patches:
            ckpt_suffix = f"{NUM_SYNTHETIC_CLASSES}class_patched"
        else:
            ckpt_suffix = "10class"
        ckpt_path = f'./logs/mnist_mlp_8neurons_{ckpt_suffix}.pt'
        os.makedirs('./logs', exist_ok=True)
        torch.save({'model': model.state_dict()}, ckpt_path)
        print(f"[INFO] Model saved to {ckpt_path}")
        
        return model, args.train_epochs, validation_loader # Return model and number of epochs trained


def plot_hessian_mnist(model, device, train_images, train_labels, args, use_patches=False, trained_epochs=None, validation_accuracy=None):
    """ Compute and plot Hessian spectrum for MNIST model. """
    from contextlib import nullcontext
    
    ctx = nullcontext()
    
    # Get all weight layers
    sample_layer = [n for n, p in model.named_parameters() if p.requires_grad and "weight" in n]

    print(f"[INFO] Number of sampled layers: {len(sample_layer)}")
    for name in sample_layer:
        print(f"  - {name}")
    
    # Add classes to output directory name
    if use_patches:
        comment = f"{args.outdir}_{NUM_SYNTHETIC_CLASSES}class_patched"
    else:
        comment = f"{args.outdir}_10class"
    
    # Determine checkpoint iteration label - use training epochs if available
    # If loaded from checkpoint, use train_epochs from args (assumes checkpoint was trained with those epochs)
    # If just trained, use the actual epochs trained
    if trained_epochs is not None:
        ckpt_iteration = trained_epochs
    else:
        # Model loaded from checkpoint - use train_epochs as the label (trained model)
        ckpt_iteration = args.train_epochs
    
    # Custom MNIST Hessian class
    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        hessian = MNISTHessian(
            model=model,
            train_images=train_images,
            train_labels=train_labels,
            ckpt_iteration=ckpt_iteration,
            batch_size=args.batch_size,
            ctx=ctx,
            use_minibatch=True,
            gradient_accumulation_steps=args.grad_accum_steps,
            device=device,
            sample_layer=sample_layer,
            comment=comment,
            m=args.m,
            num_v=args.num_v,
        )
        
        print("[INFO] Computing Hessian spectrum...")
        hessian.get_spectrum(layer_by_layer=True)
        
        print("[INFO] Loading / plotting Hessian curves...")
        hessian.load_curve(layer_by_layer=True)
    
    print("[INFO] Hessian spectrum and plots completed.")
    print(f"[INFO] Validation Accuracy: {validation_accuracy:.2f}%")


def main():
    args = get_args()
    device = get_device()

    use_patches = args.use_patches
    
    model, trained_epochs, validation_loader = load_or_train_model(device, args, use_patches=use_patches)

    # Accuracy on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in validation_loader:
            batch_images = batch_images.float().to(device)
            batch_labels = batch_labels.long().to(device)
            logits, _ = model(batch_images, batch_labels)
            pred = logits.argmax(dim=1)
            correct += pred.eq(batch_labels).sum().item()
            total += batch_labels.size(0)
    
    validation_accuracy = 100.0 * correct / total
    
    # Load MNIST training data
    train_dataset, _ = get_mnist_data(num_samples=args.num_samples, validation_size=args.validation_size, use_patches=use_patches)
    if use_patches:
        mode_str = f"{NUM_SYNTHETIC_CLASSES}-class synthetic (patches at {NUM_POSITIONS} positions)"
    else:
        mode_str = "10-class"
    print(f"[INFO] Loaded {len(train_dataset)} MNIST training samples ({mode_str} classification)")
    
    # Extract numpy arrays for plot_hessian_mnist
    train_images, train_labels = [], []
    for img, label in train_dataset:
        train_images.append(img.numpy())
        train_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    
    # Compute Hessian spectrum
    plot_hessian_mnist(model, device, train_images, train_labels, args, use_patches=use_patches, trained_epochs=trained_epochs, validation_accuracy=validation_accuracy)


if __name__ == "__main__":
    main()
