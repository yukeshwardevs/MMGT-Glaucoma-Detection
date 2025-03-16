# Glaucoma Detection Using Deep Learning: Research Background

Dataset link: https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets

## Key Research Papers and Findings

1. **ViT for Medical Image Analysis**
   - "Vision Transformers for Medical Image Classification: A Review" (2023)
   - Key finding: Vision Transformers show superior performance in medical image analysis due to their ability to capture long-range dependencies.

2. **Self-Supervised Learning in Medical Imaging**
   - "Self-supervised Visual Feature Learning for Clinical Image Analysis" (Nature Biomedical Engineering, 2021)
   - Implementation: Our SimCLR approach is based on this paper's findings showing that self-supervised pre-training improves performance on limited medical datasets.

3. **Glaucoma-Specific Research**
   - "Automated Glaucoma Detection Using Deep Learning: A Review" (Computerized Medical Imaging and Graphics, 2022)
   - Key techniques implemented:
     - CLAHE preprocessing for better feature visibility
     - Attention mechanism for focusing on relevant optic disc regions
     - Multi-scale feature extraction

## Implementation Details

### Model Architecture
Our implementation combines several proven approaches:

1. **Preprocessing Pipeline**
   - CLAHE enhancement for better contrast
   - Standard normalization and resizing
   - Data augmentation techniques validated by clinical studies

2. **Vision Transformer Backbone**
   - Base ViT architecture with 16x16 patch size
   - Modified attention mechanism for medical imaging
   - Additional skip connections for better gradient flow

3. **Self-Supervised Learning**
   - SimCLR framework for representation learning
   - Contrastive loss with temperature scaling
   - Custom augmentation pipeline for medical images

### Performance Metrics
Based on recent literature, we evaluate our model using:
- ROC-AUC (standard in medical imaging)
- Confusion matrix for clinical relevance
- Attention visualization for interpretability

### Clinical Validation
Our approach follows guidelines from:
- "Clinical Validation of AI in Ophthalmology" (JAMA Ophthalmology, 2023)
- "Standards for AI in Medical Imaging" (Radiology, 2022)

## Future Improvements

1. **Model Enhancements**
   - Integration of clinical metadata
   - Multi-modal learning with OCT images
   - Uncertainty quantification

2. **Clinical Integration**
   - Real-time processing capabilities
   - Integration with existing healthcare systems
   - Patient-specific risk assessment


# **Running the Streamlit Application**

Follow these steps to set up, train, and deploy the Streamlit application using Docker.

## **1. Clone or Download the Repository**
- Clone the repository using:
  ```bash
  git clone https://github.com/yukeshwardevs/MMGT-Glaucoma-Detection
  ```
  *OR*  
- Download and extract the repository files to your local system.

## **2. Prepare the Dataset**
- Download the required dataset.
- Move the dataset files to the active directory (where the project is located).

## **3. Update File Paths**
- Open `model.py` and `app.py` in a code editor.
- Replace the file paths with your active directory paths for the dataset and model.

## **4. Install Dependencies**
- Open a terminal in the active directory and install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

## **5. Train the Model**
- Run the following command to train the model:
  ```bash
  python3 model.py
  ```
- Once training is complete, the trained model will be saved in the active directory.

## **6. Update the Model Path in `app.py`**
- Copy the path of the saved model and update the corresponding variable in `app.py`.

## **7. Build the Docker Image**
- Run the following command to build the Docker image:
  ```bash
  docker build -f DockerFile -t mmgt-glaucoma .
  ```

## **8. Run the Docker Container**
- Start the application using:
  ```bash
  docker run -p 8501:8501 mmgt-glaucoma
  ```

## **9. Access the Application**
- Once the container is running, open a web browser and go to:
  ```
  http://localhost:8501/
  ```
- The Streamlit app should now be live and ready for use! 🎉


## References

1. Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
2. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations"
3. Wang et al. (2022). "Automated Glaucoma Screening Using Deep Learning"
4. Li et al. (2021). "Medical Image Analysis using Vision Transformers"
5. Zhang et al. (2023). "Self-supervised Learning for Medical Image Analysis"
