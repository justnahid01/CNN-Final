      ↓
CNN Model
```

### Why Preprocessing is Needed

Images can come in different sizes and formats. A CNN needs the input images to have a consistent format.

The preprocessing step helps to:

- Make all images the same size.
- Convert images into PyTorch tensors.
- Keep pixel values in a suitable range.
- Make training more stable.
- Make custom images compatible with the trained model.

The same preprocessing pipeline is applied to the custom smartphone images before prediction.

### Image Resizing

The images are resized to the required input size before being passed to the CNN.

```python
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
```

This makes sure that all images have the same dimensions.

### Converting Images to Tensor

The images are converted into PyTorch tensors using:

```python
transforms.ToTensor()
```

This changes the normal image data into a numerical format that PyTorch can process.

### Normalization

The images are normalized before being passed to the model.

```python
transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)
```

The same normalization values are used for the custom images so that their input format matches the training data.

### Complete Transform Pipeline

The preprocessing pipeline is:

```python
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])
```

---

## Custom Smartphone Images

As required by the assignment, 10 custom images were captured using a smartphone.

The images contain:

- 3 Paper images
- 4 Rock images
- 3 Scissors images

The custom images were stored in the project's GitHub repository and loaded automatically by the notebook.

Before prediction, the custom images go through the same preprocessing pipeline used for the model.

```python
image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)
```

The `unsqueeze(0)` operation adds the batch dimension required by the CNN.

### Custom Image Workflow

```text
Smartphone Image
       ↓
Resize
       ↓
ToTensor
       ↓
Normalize
       ↓
Add Batch Dimension
       ↓
Trained CNN
       ↓
Predicted Class
       ↓
Confidence Score
```

---

## Reducing Unwanted Visual Information

The assignment requires custom objects to be photographed on a simple background to reduce unwanted visual information.

A simple background helps the model focus on the hand gesture instead of unrelated objects.

However, the custom smartphone images still have some differences from the standard dataset, such as:

- Background
- Lighting
- Camera angle
- Hand position
- Image composition

These differences are important when comparing standard test accuracy with custom image accuracy.

---

## DataLoader

PyTorch `DataLoader` was used to load the training and validation data in batches.

A typical setup is:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)
```

The training data is shuffled so that the model does not always receive the images in the same order.

---

# CNN Architecture

The model was created using PyTorch and inherits from `nn.Module`.

The CNN contains:

- `nn.Conv2d` for feature extraction
- `nn.ReLU()` for activation
- `nn.MaxPool2d()` for reducing feature-map size
- `nn.Linear()` for classification

The general architecture is:

```text
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Max Pooling
     ↓
Convolution
     ↓
ReLU
     ↓
Max Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Output Layer
     ↓
Paper / Rock / Scissors
```

CNNs are useful for this task because they can learn visual patterns such as edges, shapes, fingers, and the overall hand structure.

---

# Model Training

The model was trained for **15 epochs**.

For each epoch, the model processed the training images in batches and updated its parameters based on the prediction error.

## Loss Function

Cross-Entropy Loss was used because this is a multi-class classification problem.

```python
criterion = nn.CrossEntropyLoss()
```

## Optimizer

The Adam optimizer was used to update the model parameters.

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

## Training Loop

The main training steps are:

```python
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

The process is repeated for every batch and for all 15 epochs.

---

# Training Results

The complete training results are shown below.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.9222 | 54.32% | 0.3952 | 88.10% |
| 2 | 0.2129 | 93.11% | 0.0523 | 99.21% |
| 3 | 0.0665 | 98.21% | 0.0449 | 98.81% |
| 4 | 0.0494 | 98.51% | 0.0097 | 99.80% |
| 5 | 0.0327 | 99.01% | 0.0068 | 99.80% |
| 6 | 0.0177 | 99.55% | 0.0043 | 100.00% |
| 7 | 0.0190 | 99.65% | 0.0024 | 100.00% |
| 8 | 0.0135 | 99.75% | 0.0014 | 100.00% |
| 9 | 0.0090 | 99.80% | 0.0016 | 100.00% |
| 10 | 0.0103 | 99.60% | 0.0015 | 100.00% |
| 11 | 0.0082 | 99.75% | 0.0010 | 100.00% |
| 12 | 0.0040 | 99.90% | 0.0010 | 100.00% |
| 13 | 0.0033 | 99.95% | 0.0014 | 100.00% |
| 14 | 0.0109 | 99.65% | 0.0080 | 100.00% |
| 15 | 0.0159 | 99.55% | 0.0042 | 100.00% |

The model improved very quickly during the first few epochs. Training accuracy increased from **54.32%** in the first epoch to **93.11%** in the second epoch.

The validation accuracy reached **100.00%** at epoch 6 and remained at 100% until the end of training.

---

# Training History

## Loss

![Training and Validation Loss](Training%20vs%20validation%20loss.png)

The loss decreased rapidly during the first few epochs. This shows that the model was learning the main visual patterns from the training data.

## Accuracy

![Training and Validation Accuracy](Training%20vs%20validation%20Accuracy.png)

The accuracy increased quickly during training. The validation accuracy reached 100% from epoch 6.

---

# Final Model Results

The main results of the project are:

| Measurement | Result |
|---|---:|
| Training Samples | 2016 |
| Validation Samples | 504 |
| Standard Test Samples | 372 |
| Custom Images | 10 |
| Training Epochs | 15 |
| Best Validation Accuracy | **100.00%** |
| Standard Test Accuracy | **89.52%** |
| Custom Image Accuracy | **40.00%** |

---

# Saving the Model

After training, the model state dictionary was saved using PyTorch.

```python
torch.save(
    model.state_dict(),
    "RPS_CNN.pth"
)
```

The saved `.pth` file contains the learned parameters of the CNN.

The model can later be loaded without training from the beginning.

```python
model.load_state_dict(
    torch.load(
        "RPS_CNN.pth",
        map_location=device
    )
)

model.eval()
```

---

# Standard Test Evaluation

The trained model was evaluated using **372 standard test images**.

The model correctly classified **333 images**.

Therefore:

```text
Accuracy = (333 / 372) × 100
         = 89.52%
```

The final standard test accuracy was:

**89.52%**

---

# Confusion Matrix

![Confusion Matrix](Confusion%20matrix.png)

The confusion matrix shows the predictions for each class.

The results were:

| Actual Class | Correct Predictions |
|---|---:|
| Paper | 124 |
| Rock | 97 |
| Scissors | 112 |

The model performed particularly well on the Paper class.

Some Rock images were classified as Paper or Scissors, while some Scissors images were classified as Paper.

---

# Classification Report

![Classification Report](Classification%20Report.png)

The classification report gives precision, recall, and F1-score for each class.

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Paper | 0.81 | 1.00 | 0.89 |
| Rock | 1.00 | 0.78 | 0.88 |
| Scissors | 0.93 | 0.90 | 0.91 |

The overall test accuracy was **89.52%**.

Paper had a recall of 1.00, meaning all actual Paper images in the test set were identified as Paper.

Rock had a precision of 1.00, meaning the Rock predictions made by the model were correct, although some actual Rock images were classified as another class.

---

# Custom Smartphone Testing

The trained model was tested on 10 custom smartphone images.

The images were:

- `paper1.jpg`
- `paper2.jpg`
- `paper3.jpg`
- `rock1.jpg`
- `rock2.jpg`
- `rock3.jpg`
- `rock4.jpg`
- `scissors1.jpg`
- `scissors2.jpg`
- `scissors3.jpg`

## Custom Predictions

| File | True Class | Predicted Class | Confidence |
|---|---|---|---:|
| paper1.jpg | Paper | Rock | 83.70% |
| paper2.jpg | Paper | Rock | 77.20% |
| paper3.jpg | Paper | Rock | 97.21% |
| rock1.jpg | Rock | Rock | 89.10% |
| rock2.jpg | Rock | Rock | 99.69% |
| rock3.jpg | Rock | Rock | 99.99% |
| rock4.jpg | Rock | Rock | 97.93% |
| scissors1.jpg | Scissors | Rock | 97.66% |
| scissors2.jpg | Scissors | Paper | 87.49% |
| scissors3.jpg | Scissors | Rock | 97.95% |

---

# Custom Prediction Gallery

![Custom Predictions](custom%2010%20img%20prediction.png)

The model correctly classified all four Rock images.

However:

- All three Paper images were predicted as Rock.
- Two Scissors images were predicted as Rock.
- One Scissors image was predicted as Paper.

Therefore:

```text
Correct Predictions = 4
Total Images = 10

Custom Accuracy = (4 / 10) × 100
                = 40.00%
```

The final custom image accuracy was:

**40.00%**

---

# Why Is Custom Accuracy Lower?

The difference between the standard test accuracy and custom accuracy is important.

```text
Best Validation Accuracy : 100.00%
Standard Test Accuracy   : 89.52%
Custom Image Accuracy    : 40.00%
```

The standard test images come from the same general dataset used during model development.

The smartphone images are different in several ways:

- Background
- Lighting
- Camera angle
- Hand position
- Image composition

Because of these differences, the model did not perform as well on the custom images.

This shows that high accuracy on a standard dataset does not always mean that the model will perform equally well in a real-world environment.

---

# Visual Error Analysis

Three incorrectly classified images from the standard test set were selected for analysis.

![Visual Error Analysis](Visual%20Error%20Analysis.png)

The examples show cases where the model confused different hand gestures.

Some possible reasons include:

- Similar hand shapes
- Different hand positions
- Lighting differences
- Background information
- Camera angle
- Image quality

Looking at incorrect predictions helps identify where the model needs improvement.

---

# Project Workflow

The complete project workflow is:

```text
Standard RPS Dataset
        ↓
Data Preprocessing
        ↓
Training / Validation Data
        ↓
DataLoader
        ↓
CNN Model
        ↓
Training for 15 Epochs
        ↓
Validation
        ↓
Save Model (.pth)
        ↓
Standard Test
        ↓
Confusion Matrix
        ↓
Custom Smartphone Images
        ↓
Same Preprocessing
        ↓
CNN Prediction
        ↓
Softmax Probabilities
        ↓
Confidence Scores
        ↓
Custom Accuracy
        ↓
Visual Error Analysis
```

---

# Project Structure

The project repository contains the main project files:

```text
CNN-Final/
│
├── dataset/
│   ├── paper1.jpg
│   ├── paper2.jpg
│   ├── paper3.jpg
│   ├── rock1.jpg
│   ├── rock2.jpg
│   ├── rock3.jpg
│   ├── rock4.jpg
│   ├── scissors1.jpg
│   ├── scissors2.jpg
│   └── scissors3.jpg
│
├── model/
│   └── RPS_CNN.pth
│
├── RPS_CNN.ipynb
│
└── README.md
```

---

# Automatic Data Loading

The notebook is designed to retrieve the project files from GitHub.

The repository can be cloned using:

```bash
git clone https://github.com/justnahid01/CNN-Final.git
```

This allows the notebook to access the custom images without manually uploading them during runtime.

The standard dataset is loaded separately using the dataset-loading functionality provided by the project.

---

# How to Run

## Google Colab

Open the notebook:

https://colab.research.google.com/drive/1BHMtfzsph6hds9jqm50E3Mt9iCWjQdP8?usp=sharing

Then run the notebook from the beginning.

## GitHub

Clone the repository:

```bash
git clone https://github.com/justnahid01/CNN-Final.git
```

Then open the notebook using Google Colab or Jupyter Notebook.

---

# Requirements Covered

This project covers the main requirements of the CNN assignment:

- [x] Standard Rock-Paper-Scissors dataset
- [x] PyTorch CNN
- [x] `nn.Module`
- [x] Convolution layers
- [x] ReLU activation
- [x] Max Pooling
- [x] Fully Connected layers
- [x] Image resizing
- [x] Tensor conversion
- [x] Image normalization
- [x] DataLoader
- [x] Cross-Entropy Loss
- [x] Adam optimizer
- [x] Training loop
- [x] 15 training epochs
- [x] Saved `.pth` model
- [x] Standard test evaluation
- [x] Confusion matrix
- [x] Training and validation loss graph
- [x] Training and validation accuracy graph
- [x] 10 custom smartphone images
- [x] Custom prediction gallery
- [x] Prediction confidence scores
- [x] Custom accuracy calculation
- [x] Visual error analysis

---

# Future Improvements

The project can be improved by:

- Collecting more custom smartphone images.
- Using more varied backgrounds.
- Taking images under different lighting conditions.
- Collecting images from different people.
- Using data augmentation.
- Increasing the variety of training images.
- Trying transfer learning with models such as ResNet or MobileNet.

More diverse training data would likely help the model perform better on real-world images.

---

# Conclusion

This project demonstrates a complete CNN image classification workflow using PyTorch.

The model was trained to recognize Paper, Rock, and Scissors gestures. It achieved **100.00% validation accuracy** and **89.52% standard test accuracy**.

When tested on 10 custom smartphone images, the model achieved **40.00% accuracy**.

The difference between standard test performance and custom performance shows that real-world images can be more challenging because their background, lighting, camera angle, and composition can be different from the training data.

Overall, the project provided practical experience with image preprocessing, CNN architecture, model training, evaluation, custom image prediction, confidence scores, and error analysis.

---

## Project Links

**GitHub:**  
https://github.com/justnahid01/CNN-Final

**Google Colab:**  
https://colab.research.google.com/drive/1BHMtfzsph6hds9jqm50E3Mt9iCWjQdP8?usp=sharing

---

**Author:** Nahid Hasan  
**Roll:** 220112
