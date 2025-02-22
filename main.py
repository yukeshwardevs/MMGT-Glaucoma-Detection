import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from timm import create_model
import cv2
import os
from PIL import Image

# Dataset class remains the same
class FundusDataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image_path = os.path.join(self.image_folder, image_name)
        image = preprocess_image(image_path)
        image = torch.tensor(image).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, label

# Preprocessing function remains the same
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image

# Modified SimCLR model with classification head
class SimCLR(nn.Module):
    def __init__(self, base_model='vit_base_patch16_224', out_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = create_model(base_model, pretrained=True, num_classes=0)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        # Add classification head
        self.classification_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output single value for binary classification
        )
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        projections = self.projection_head(features)
        if return_features:
            return projections
        classifications = self.classification_head(projections)
        return classifications

# Modified training function
def train_model(model, dataloader, epochs=10, lr=1e-4, save_path="model.pth"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Get model outputs (now properly shaped for binary classification)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Define transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])
    
    # Setup dataset and dataloader
    csv_path = "/kaggle/input/glaucoma-datasets/G1020/G1020.csv"  # Update with actual path
    image_folder = "/kaggle/input/glaucoma-datasets/G1020/Images_Cropped/img"   # Update with actual path
    dataset = FundusDataset(csv_path, image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model and train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr = SimCLR().to(device)
    
    # Train the model
    train_model(simclr, dataloader, epochs=5, save_path="simclr_model.pth")
    
    # Extract features for clustering (if needed)
    def extract_features(model, dataloader):
        model.eval()
        features_list = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                features = model(images, return_features=True)
                features_list.append(features.cpu().numpy())
        return np.concatenate(features_list)
    
    # Extract features and perform clustering
    features = extract_features(simclr, dataloader)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    pseudo_labels = dbscan.fit_predict(features)
