import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

# First define the train/test split ratio
def create_data_loaders(csv_path, image_folder, batch_size=8, train_ratio=0.8):
    """
    Create train and test data loaders
    
    Args:
        csv_path (str): Path to CSV file with image names and labels
        image_folder (str): Path to folder containing images
        batch_size (int): Batch size for data loaders
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
    
    Returns:
        train_loader, test_loader
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])
    
    # Test transform doesn't need augmentation
    test_transform = transforms.Compose([])
    
    # Create full dataset
    full_dataset = FundusDataset(csv_path, image_folder, transform=None)
    
    # Calculate lengths for split
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Add transforms to the splits
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

# Modified main execution code
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths
    csv_path = "/kaggle/input/glaucoma-datasets/G1020/G1020.csv"  # Update with your actual path
    image_folder = "/kaggle/input/glaucoma-datasets/G1020/Images_Cropped/img"  # Update with your actual path
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        csv_path=csv_path,
        image_folder=image_folder,
        batch_size=8,
        train_ratio=0.8
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    # Initialize model
    model = SimCLR().to(device)
    
    # Train model
    train_model(model, train_loader, epochs=5, save_path="simclr_model.pth")
    
    # Initialize visualizer with test loader
    visualizer = GlaucomaVisualizer(model, test_loader, device)
    
    # Generate visualizations
    visualizer.plot_roc_curve()
    visualizer.plot_confusion_matrix()
    
    # Evaluate model
    accuracy, probs, labels = evaluate_model(model, test_loader, device)
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    # Optional: Visualize attention maps for a specific image
    example_image_path = "/kaggle/input/glaucoma-datasets/G1020/Images_Cropped/image_2530.jpg"  # Update with actual image path
    visualizer.visualize_attention_maps(example_image_path)

# Function to check data distribution
def check_data_distribution(csv_path):
    """Print distribution of labels in dataset"""
    df = pd.read_csv(csv_path)
    label_column = df.columns[1]  # Assuming second column contains labels
    
    # Get distribution
    distribution = df[label_column].value_counts()
    total = len(df)
    
    print("\nData Distribution:")
    for label, count in distribution.items():
        percentage = (count / total) * 100
        print(f"Class {label}: {count} images ({percentage:.2f}%)")
    
    return distribution
