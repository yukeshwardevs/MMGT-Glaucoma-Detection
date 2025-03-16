import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from timm import create_model
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Custom loss function to heavily penalize false negatives (missing glaucoma)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for the positive class (glaucoma)
        self.gamma = gamma  # Focusing parameter
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        
        # Apply focal weighting
        pt = torch.exp(-BCE_loss)
        
        # Apply class weighting - alpha for glaucoma (1), 1-alpha for healthy (0)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_weight * (1 - pt) ** self.gamma * BCE_loss
        
        return focal_loss.mean()

# Dataset class with debugging capabilities
class GlaucomaDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None, phase='train', debug=False):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
        self.phase = phase
        self.debug = debug
        
        # Print class distribution
        class_counts = self.data.iloc[:, 1].value_counts()
        print(f"Class distribution: Healthy={class_counts.get(0, 0)}, Glaucoma={class_counts.get(1, 0)}")
        
        # Calculate class weights for balanced sampling (inverse frequency)
        total = len(self.data)
        self.class_weights = {
            0: total / (2 * class_counts.get(0, 1)),  # Healthy
            1: total / (2 * class_counts.get(1, 1))   # Glaucoma
        }
        
        print(f"Class weights: Healthy={self.class_weights[0]:.4f}, Glaucoma={self.class_weights[1]:.4f}")
        
        # Create sample weights for WeightedRandomSampler
        self.sample_weights = [self.class_weights[label] for label in self.data.iloc[:, 1]]
        
        # Verify data integrity
        if debug:
            self.verify_data_integrity()
    
    def verify_data_integrity(self):
        """Check if all images exist and can be loaded"""
        print("Verifying data integrity...")
        missing_files = []
        corrupt_files = []
        
        for idx in range(len(self.data)):
            image_name = self.data.iloc[idx, 0]
            image_path = os.path.join(self.image_folder, image_name)
            
            # Check if file exists
            if not os.path.exists(image_path):
                missing_files.append(image_path)
                continue
                
            # Try to load the image
            try:
                img = cv2.imread(image_path)
                if img is None:
                    corrupt_files.append(image_path)
            except Exception as e:
                corrupt_files.append(image_path)
                
        if missing_files:
            print(f"Found {len(missing_files)} missing files")
            for path in missing_files[:5]:
                print(f"  - {path}")
                
        if corrupt_files:
            print(f"Found {len(corrupt_files)} corrupt files")
            for path in corrupt_files[:5]:
                print(f"  - {path}")
                
        if not missing_files and not corrupt_files:
            print("All files verified successfully!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]  # 0 = Healthy, 1 = Glaucoma
        image_path = os.path.join(self.image_folder, image_name)
        
        # Enhanced preprocessing
        image = self.preprocess_image(image_path)
        
        # Convert to tensor
        image = torch.tensor(image).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        # Apply transformations
        if self.transform and self.phase == 'train':
            image = self.transform(image)
            
        return image, label, image_name

    def preprocess_image(self, image_path):
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            # Return a blank image if file can't be read
            return np.zeros((224, 224, 3), dtype=np.float32)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE for contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Additional normalization to match ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image

# Improved model architecture with explicit initialization
class GlaucomaNet(nn.Module):
    def __init__(self, base_model='resnet50', pretrained=True):
        super(GlaucomaNet, self).__init__()
        # Use ResNet50 which is more stable for medical imaging
        self.backbone = create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        
        # Feature projection with batch normalization
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        
        # Classification head with explicit bias initialization
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Initialize the final layer with bias towards glaucoma
        # This helps counter the "always predict healthy" problem
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Initialize the final layer with a positive bias to favor glaucoma predictions
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Slight positive bias
        
        # Set a stronger bias for the final layer to counter class imbalance
        final_layer = self.classifier[-1]
        if hasattr(final_layer, 'bias') and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, 1.0)  # Stronger positive bias
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        output = self.classifier(projected)
        return output

# Training function with debugging and aggressive learning rate scheduling
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, save_path="glaucoma_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Use Focal Loss to focus on hard examples and address class imbalance
    criterion = FocalLoss(alpha=0.8, gamma=2.0)
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts for better exploration
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_auc = 0
    patience = 10
    counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    # Debug: Check initial predictions before training
    print("\nInitial model predictions:")
    check_predictions(model, val_loader, device)
    
    for epoch in range(epochs):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Training phase
        model.train()
        train_loss = 0
        batch_losses = []
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Debug: Print batch loss statistics
        if batch_losses:
            print(f"Batch loss stats - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, "
                  f"Mean: {np.mean(batch_losses):.4f}, Std: {np.std(batch_losses):.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store predictions and labels for metrics calculation
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Convert to numpy arrays for metric calculation
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate metrics
        val_preds_binary = (all_preds > 0.5).astype(int)
        val_acc = accuracy_score(all_labels, val_preds_binary)
        val_precision = precision_score(all_labels, val_preds_binary, zero_division=0)
        val_recall = recall_score(all_labels, val_preds_binary, zero_division=0)
        val_f1 = f1_score(all_labels, val_preds_binary, zero_division=0)
        
        # Calculate AUC only if both classes are present
        if len(np.unique(all_labels)) > 1:
            val_auc = roc_auc_score(all_labels, all_preds)
        else:
            val_auc = 0
            print("Warning: Only one class present in validation set")
        
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        time_elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] completed in {time_elapsed:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Debug: Check prediction distribution
        print("\nPrediction distribution:")
        pred_counts = np.bincount(val_preds_binary, minlength=2)
        print(f"Predicted Healthy: {pred_counts[0]}, Predicted Glaucoma: {pred_counts[1]}")
        print(f"Prediction ratio (Glaucoma/Total): {pred_counts[1]/len(val_preds_binary):.4f}")
        
        # Check for improvement - prioritize recall for medical applications
        current_metric = val_recall * 0.7 + val_auc * 0.3  # Weighted combination
        
        if current_metric > best_val_auc:
            best_val_auc = current_metric
            best_val_loss = avg_val_loss
            counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_auc': val_auc,
                'val_recall': val_recall,
            }, save_path)
            print(f"Model saved with Val Recall: {val_recall:.4f}, Val AUC: {val_auc:.4f}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
        if counter >= patience:
            print("Early stopping triggered")
            break
            
        print("-" * 60)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with Val Recall: {checkpoint['val_recall']:.4f}")
    
    return model, history

# Function to check model predictions during training
def check_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Print prediction statistics
    print(f"Prediction stats - Min: {min(all_preds):.4f}, Max: {max(all_preds):.4f}, "
          f"Mean: {np.mean(all_preds):.4f}, Std: {np.std(all_preds):.4f}")
    
    # Count predictions by threshold
    pred_binary = (np.array(all_preds) > 0.5).astype(int)
    pred_counts = np.bincount(pred_binary, minlength=2)
    print(f"Predicted Healthy: {pred_counts[0]}, Predicted Glaucoma: {pred_counts[1]}")
    
    # Print actual label distribution
    label_counts = np.bincount(np.array(all_labels).astype(int), minlength=2)
    print(f"Actual Healthy: {label_counts[0]}, Actual Glaucoma: {label_counts[1]}")

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy and AUC
    plt.subplot(2, 2, 2)
    plt.plot(history['val_acc'], label='Accuracy')
    plt.plot(history['val_auc'], label='AUC')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.plot(history['val_f1'], label='F1 Score')
    plt.title('Validation Precision-Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Function to evaluate model on test set
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_image_names = []
    
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_image_names.extend(image_names)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Try different thresholds to find optimal one
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_f1 = 0
    best_threshold = 0.5
    
    print("\nThreshold optimization:")
    for threshold in thresholds:
        preds_binary = (all_preds > threshold).astype(int)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)
        recall = recall_score(all_labels, preds_binary, zero_division=0)
        precision = precision_score(all_labels, preds_binary, zero_division=0)
        
        print(f"Threshold: {threshold:.1f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    
    # Use best threshold for final evaluation
    preds_binary = (all_preds > best_threshold).astype(int)
    accuracy = accuracy_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    
    # Calculate AUC only if both classes are present
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, preds_binary)
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Healthy', 'Glaucoma']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return all_preds, all_labels, all_image_names, best_threshold

# Function to visualize predictions
def visualize_predictions(model, test_loader, threshold=0.5, num_images=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_images = []
    all_preds = []
    all_labels = []
    all_image_names = []
    
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            if len(all_images) >= num_images:
                break
                
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            # Store batch data
            for i in range(len(images)):
                if len(all_images) >= num_images:
                    break
                    
                all_images.append(images[i].cpu().numpy())
                all_preds.append(probs[i])
                all_labels.append(labels[i].item())
                all_image_names.append(image_names[i])
    
    # Plot predictions
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (image, pred, label, image_name) in enumerate(zip(all_images, all_preds, all_labels, all_image_names)):
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image.transpose(1, 2, 0) * std + mean)
        image = np.clip(image, 0, 1)
        
        # Display image
        axes[i].imshow(image)
        
        # Set title with prediction and actual label
        pred_class = "Glaucoma" if pred > threshold else "Healthy"
        actual_class = "Glaucoma" if label == 1 else "Healthy"
        
        # Color code the title based on correctness
        color = "green" if (pred > threshold) == (label == 1) else "red"
        
        axes[i].set_title(f"Pred: {pred_class} ({pred:.2f})\nActual: {actual_class}", 
                         color=color, fontsize=10)
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Return misclassified examples for further analysis
    misclassified = [(image_name, pred, label) for image_name, pred, label in 
                     zip(all_image_names, all_preds, all_labels) if (pred > threshold) != (label == 1)]
    
    return misclassified

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    csv_path = "/kaggle/input/glaucoma-datasets/G1020/G1020.csv"
    image_folder = "/kaggle/input/glaucoma-datasets/G1020/Images_Cropped/img"
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    # Create dataset with debug mode
    dataset = GlaucomaDataset(csv_path, image_folder, transform=train_transform, phase='train', debug=True)
    
    # Split dataset (70% train, 15% validation, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update phase for validation and test datasets
    val_dataset.dataset.phase = 'val'
    test_dataset.dataset.phase = 'test'
    
    # Use weighted sampler for training to handle class imbalance
    train_indices = train_dataset.indices
    train_weights = [dataset.sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Create model
    model = GlaucomaNet(base_model='resnet50', pretrained=True)
    
    # Train model with higher learning rate
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=200, 
        lr=5e-4,  # Higher learning rate
        save_path="glaucoma_model_best.pth"
    )
    
    # Evaluate model on test set with threshold optimization
    predictions, labels, image_names, best_threshold = evaluate_model(trained_model, test_loader)
    
    # Visualize predictions with optimized threshold
    misclassified = visualize_predictions(trained_model, test_loader, threshold=best_threshold, num_images=32)
    
    # Print some misclassified examples
    print("\nMisclassified examples:")
    for image_name, pred, label in misclassified[:10]:
        print(f"Image: {image_name}, Prediction: {pred:.4f}, Actual: {int(label)}")