import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from timm import create_model
import time

# Set page configuration
st.set_page_config(
    page_title="Glaucoma Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .result-text-healthy {
        font-size: 1.8rem;
        color: #4CAF50;
        font-weight: bold;
    }
    .result-text-glaucoma {
        font-size: 1.8rem;
        color: #F44336;
        font-weight: bold;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Model definition
class GlaucomaNet(nn.Module):
    def __init__(self, base_model='resnet50', pretrained=False):
        super(GlaucomaNet, self).__init__()
        self.backbone = create_model(base_model, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        
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
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        projected = self.projection(features)
        output = self.classifier(projected)
        return output

# Image preprocessing function
def preprocess_image(image):
    """Preprocess image for model input"""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Resize
    image = cv2.resize(image, (224, 224))
    
    # Save original for display
    display_image = image.copy()
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Additional normalization to match ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor
    image_tensor = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
    
    return image_tensor, display_image

# Load model
@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GlaucomaNet(base_model='resnet50').to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            val_auc = checkpoint.get('val_auc', 'unknown')
            st.sidebar.info(f"Model loaded from epoch: {epoch}, Validation AUC: {val_auc}")
        else:
            model.load_state_dict(checkpoint)
            st.sidebar.info("Model loaded successfully")
            
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to make prediction
def predict(model, image_tensor, device, threshold=0.5):
    """Make prediction using the model"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > threshold else 0
            return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Function to visualize the result
def visualize_result(image, prediction, probability, threshold=0.5):
    """Visualize the prediction result"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the image
    ax.imshow(image)
    ax.axis('off')
    
    # Add prediction overlay
    if prediction == 1:  # Glaucoma
        # Add red tint for glaucoma
        overlay = np.zeros_like(image, dtype=np.uint8)
        overlay[:, :, 0] = 150  # Red channel
        alpha = 0.3
        highlighted = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        ax.imshow(highlighted)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(5)
    
    # Return the figure
    return fig

# Function to get confidence level text and color
def get_confidence_level(probability, threshold=0.5):
    """Get confidence level text and color based on probability"""
    if probability > threshold:
        # Glaucoma prediction
        if probability > 0.8:
            return "High", "confidence-high"
        elif probability > 0.65:
            return "Medium", "confidence-medium"
        else:
            return "Low", "confidence-low"
    else:
        # Healthy prediction
        if probability < 0.2:
            return "High", "confidence-high"
        elif probability < 0.35:
            return "Medium", "confidence-medium"
        else:
            return "Low", "confidence-low"

# Main function
def main():
    # Header
    st.markdown("<h1 class='main-header'>Glaucoma Detection System</h1>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://img.freepik.com/free-vector/eye-care-concept-illustration_114360-3655.jpg", width=250)
    st.sidebar.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.sidebar.info(
        "This application uses a deep learning model to detect glaucoma from fundus images. "
        "Upload a fundus image to get a prediction."
    )
    
    # Model settings
    st.sidebar.markdown("<h2 class='sub-header'>Model Settings</h2>", unsafe_allow_html=True)
    model_path = st.sidebar.text_input("Model Path", "glaucoma_model_best.pth")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Load model
    model, device = load_model(model_path)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Upload Fundus Image</h2>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])
        
        # Sample images
        st.markdown("<h3>Or try a sample image:</h3>", unsafe_allow_html=True)
        sample_col1, sample_col2 = st.columns(2)
        
        # You can replace these with actual sample images from your dataset
        sample1 = sample_col1.button("Sample Healthy Eye")
        sample2 = sample_col2.button("Sample Glaucoma Eye")
        
        # Process sample images
        if sample1:
            # Replace with path to a sample healthy image
            image = Image.open("sample_healthy.jpg") if os.path.exists("sample_healthy.jpg") else None
            if image is None:
                # Create a placeholder image
                image = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 200)
                st.warning("Sample image not found. Using placeholder.")
            uploaded_file = "sample_healthy"
        elif sample2:
            # Replace with path to a sample glaucoma image
            image = Image.open("sample_glaucoma.jpg") if os.path.exists("sample_glaucoma.jpg") else None
            if image is None:
                # Create a placeholder image
                image = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 150)
                st.warning("Sample image not found. Using placeholder.")
            uploaded_file = "sample_glaucoma"
        
        # Information section
        st.markdown("<h2 class='sub-header'>What is Glaucoma?</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='info-text'>Glaucoma is a group of eye conditions that damage the optic nerve, "
            "which is vital for good vision. This damage is often caused by abnormally high pressure "
            "in your eye. Glaucoma is one of the leading causes of blindness for people over the age of 60. "
            "It can occur at any age but is more common in older adults.</p>",
            unsafe_allow_html=True
        )
        
        st.markdown("<h2 class='sub-header'>Signs and Symptoms</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p class='info-text'>"
            "• Patchy blind spots in peripheral or central vision<br>"
            "• Tunnel vision in advanced stages<br>"
            "• Severe headache<br>"
            "• Eye pain<br>"
            "• Nausea and vomiting<br>"
            "• Blurred vision<br>"
            "• Halos around lights<br>"
            "• Eye redness</p>",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("<h2 class='sub-header'>Detection Results</h2>", unsafe_allow_html=True)
        
        if uploaded_file is not None and model is not None:
            # Display spinner during processing
            with st.spinner("Processing image..."):
                try:
                    # Load and preprocess image
                    if isinstance(uploaded_file, str):  # Sample image already loaded
                        pil_image = image
                    else:  # Uploaded file
                        image_bytes = uploaded_file.read()
                        pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Preprocess image
                    image_tensor, display_image = preprocess_image(pil_image)
                    
                    # Make prediction
                    start_time = time.time()
                    prediction, probability = predict(model, image_tensor, device, threshold)
                    inference_time = time.time() - start_time
                    
                    if prediction is not None:
                        # Display result
                        result_fig = visualize_result(display_image, prediction, probability, threshold)
                        st.pyplot(result_fig)
                        
                        # Display prediction
                        if prediction == 1:
                            st.markdown("<h3 class='result-text-glaucoma'>Glaucoma Detected</h3>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h3 class='result-text-healthy'>Healthy Eye</h3>", unsafe_allow_html=True)
                        
                        # Display confidence
                        confidence_level, confidence_class = get_confidence_level(probability, threshold)
                        st.markdown(
                            f"<p>Probability: {probability:.2f} | Confidence: <span class='{confidence_class}'>{confidence_level}</span></p>",
                            unsafe_allow_html=True
                        )
                        
                        # Display inference time
                        st.info(f"Inference time: {inference_time*1000:.2f} ms")
                        
                        # Display interpretation
                        st.markdown("<h3>Interpretation</h3>", unsafe_allow_html=True)
                        if prediction == 1:
                            st.markdown(
                                "<p>The model has detected signs consistent with glaucoma in this fundus image. "
                                "Key indicators may include:</p>"
                                "<ul>"
                                "<li>Increased cup-to-disc ratio</li>"
                                "<li>Neuroretinal rim thinning</li>"
                                "<li>Optic nerve head changes</li>"
                                "<li>Retinal nerve fiber layer defects</li>"
                                "</ul>"
                                "<p><strong>Note:</strong> This is an automated screening tool. "
                                "Please consult with an ophthalmologist for clinical diagnosis.</p>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                "<p>The model has not detected significant signs of glaucoma in this fundus image. "
                                "The optic nerve head appears to be within normal parameters.</p>"
                                "<p><strong>Note:</strong> Regular eye check-ups are still recommended for early detection "
                                "of any eye conditions.</p>",
                                unsafe_allow_html=True
                            )
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        else:
            # Placeholder
            st.info("Please upload an image or select a sample image to get started.")
            
            # Display placeholder image
            placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "No image uploaded", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            st.image(placeholder, use_column_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center'>Glaucoma Detection System | Developed with Streamlit and PyTorch</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()