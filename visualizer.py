import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torchvision.utils import make_grid
import cv2
from scipy import ndimage

class GlaucomaVisualizer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
    def plot_roc_curve(self):
        """Plot ROC curve for model evaluation"""
        self.model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        self.model.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
    def visualize_attention_maps(self, image_path):
        """Visualize attention maps for a single image"""
        # Load and preprocess image
        image = preprocess_image(image_path)
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get attention weights from the ViT
            attentions = self.model.encoder.get_attention_map(image_tensor)
            
            # Average attention across heads
            attention_map = attentions.mean(1).mean(1)  # Average across heads and layers
            attention_map = attention_map.reshape(14, 14)  # Reshape to patch grid
            attention_map = ndimage.zoom(attention_map.cpu().numpy(), 
                                      (16, 16),  # Scale up to image size
                                      order=1)
            
            # Plot original image and attention map
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            im = ax2.imshow(attention_map, cmap='hot')
            ax2.set_title('Attention Map')
            ax2.axis('off')
            plt.colorbar(im)
            plt.show()

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_probs, all_labels

# Usage example:
if __name__ == "__main__":
    # Initialize model and load weights
    model = SimCLR().to(device)
    model.load_state_dict(torch.load("simclr_model.pth"))
    
    # Create visualizer
    visualizer = GlaucomaVisualizer(model, test_loader, device)
    
    # Generate visualizations
    visualizer.plot_roc_curve()
    visualizer.plot_confusion_matrix()
    visualizer.visualize_attention_maps("example_image.jpg")
    
    # Evaluate model
    accuracy, probs, labels = evaluate_model(model, test_loader, device)
    print(f"Model Accuracy: {accuracy:.2f}%")
