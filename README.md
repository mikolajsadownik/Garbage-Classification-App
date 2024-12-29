# Garbage-Classification-App
This is a web-based app that classifies images of garbage into six categories: **cardboard**, **glass**, **metal**, **paper**, **plastic**, and **trash**. The app uses a pre-trained neural network model and provides an easy-to-use interface for uploading and analyzing images.
 
## Features

- **Upload an image** via drag-and-drop or file browsing.
- **Displays**:
  - The predicted garbage category.
  - Confidence score for the prediction.
  - A table and bar chart of probabilities for all categories.
  - The uploaded image and the processed image.
- **Option to download** the uploaded image.

![image](https://github.com/user-attachments/assets/1f726861-0cde-42c6-94a1-0d7366c02865)
![image](https://github.com/user-attachments/assets/7f959888-f86d-48dc-8f2c-1f59a8f79553)
![image](https://github.com/user-attachments/assets/45314272-7afd-4896-bda1-af95c86712d2)

## Model Architecture
1. **Base Model**:
   - **EfficientNetB0**:
     - Pre-trained on ImageNet.
     - Excludes the top classification layer (`include_top=False`).
     - Input shape: `(224, 224, 3)` for RGB images.
     - The base model parameters are frozen (`trainable=False`) to preserve pre-trained features.
     
     **Description**: EfficientNet is a family of models designed to achieve state-of-the-art performance while minimizing computational resources. These models balance depth, width, and resolution scaling to maximize computational efficiency.

2. **Custom Classification Head**:
   - **Dropout Layer**:
     - Dropout rate: 20% (`Dropout(0.2)`) to prevent overfitting.
   - **Global Average Pooling**:
     - Reduces spatial dimensions to a vector representation.
   - **Dense Layer**:
     - Fully connected layer with `num_classes=6` output units.
     - Activation function: Softmax for multi-class classification.

---

### Dataset Preparation

1. **Dataset Splitting**:
   - Data is sourced from the `data` directory.
   - Labels are inferred from folder names (`label_mode='categorical'`).
   - Images are resized to `(224, 224)`.
   - Data is split into training (70%) and validation (30%) sets using `validation_split=0.3`.
   - A consistent split is ensured with a fixed random seed (`SEED=2137`).

2. **Batching**:
   - Batch size: 32.
   - Data is shuffled for randomness during training.

---

### Training Configuration

**Hyperparameters**:
   - Number of epochs: 26.
   - Optimizer: Adam, selected for its adaptive learning rate capabilities.
   - Loss Function: Categorical Crossentropy, suitable for multi-class classification.
   - Metrics: Accuracy to evaluate model performance.

---

### Methodology

1. **Transfer Learning**:
   - EfficientNetB0 is used as a feature extractor by freezing its weights.
   - A custom Dense classification head is added and trained on the new dataset.

2. **Training Process**:
   - The model is trained on the processed training dataset and evaluated on the validation dataset.
   - Epoch-wise updates track improvements in accuracy and loss function.

---

### Key Highlights

- **Generalization**:
  - The model leverages pre-trained ImageNet weights for robust feature extraction, even with limited data.
- **Efficiency**:
  - EfficientNetB0 provides a balanced trade-off between performance and computational efficiency.
- **Fine-tuning**:
  - Dropout and Global Average Pooling layers reduce overfitting and enhance the model's adaptability to new data.

---

### Performance on 30% validation set

- **Accuracy 90.63%**
- **Confusion Matrix**

![image](https://github.com/link-to-image-here)

