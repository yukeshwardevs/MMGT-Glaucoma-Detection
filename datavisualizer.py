import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# Define paths
csv_path = "/kaggle/input/glaucoma-datasets/G1020/G1020.csv"
image_folder = "/kaggle/input/glaucoma-datasets/G1020/Images_Cropped/img"

# Load CSV
df = pd.read_csv(csv_path)

# Ensure column names are correct
image_column = df.columns[0]  # First column (image names)
label_column = df.columns[1]  # Second column (labels)

# Get first two images labeled 1 and first two labeled 0
glaucoma_present = df[df[label_column] == 1][image_column].iloc[:2].tolist()
glaucoma_absent = df[df[label_column] == 0][image_column].iloc[:2].tolist()

# Function to display images
def display_images(image_names, title):
    plt.figure(figsize=(10, 5))
    for i, image_name in enumerate(image_names):
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display
        
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{title} {i+1} {image_name}")
    
    plt.show()

# Display images
display_images(glaucoma_present, "Glaucoma Present")
display_images(glaucoma_absent, "Glaucoma Absent")

random_samples = df.sample(n=10, random_state=42)  # Set seed for reproducibility

# Function to display images
def display_images(image_list):
    plt.figure(figsize=(15, 7))
    
    for i, (image_name, label) in enumerate(zip(image_list[image_column], image_list[label_column])):
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display
        
        plt.subplot(2, 5, i + 1)  # Arrange in 2 rows, 5 columns
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Label: {label}")
    
    plt.show()

# Display images
display_images(random_samples)
