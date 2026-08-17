# Rock-Paper-Scissors Image Classification using CNN

A PyTorch-based Convolutional Neural Network (CNN) project for classifying hand gestures into three classes:

- Paper
- Rock
- Scissors

The project follows a complete image classification workflow, starting from a standard Rock-Paper-Scissors dataset and ending with real-world testing using custom smartphone photographs.

---

## Project Overview

The main goal of this project is to build a CNN that can recognize Rock, Paper, and Scissors hand gestures from images.

The model was trained using a standard RPS dataset and then tested in two different ways:

1. **Standard Test Dataset** – 372 unseen images from the dataset.
2. **Custom Smartphone Images** – 10 photographs taken using a phone.

This allowed us to compare the model's performance on dataset images with its performance on real-world images.

---

## Student Information

**Name:** Nahid Hasan  
**Roll:** 220112

### Project Links

- **GitHub:** https://github.com/justnahid01/CNN-Final
- **Google Colab:** https://colab.research.google.com/drive/1BHMtfzsph6hds9jqm50E3Mt9iCWjQdP8?usp=sharing

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Google Colab
- GitHub

---

## Dataset

The project uses a standard Rock-Paper-Scissors image dataset.

The model contains three classes:

| Class | Label |
|---|---:|
| Paper | 0 |
| Rock | 1 |
| Scissors | 2 |

### Dataset Size

| Dataset | Images |
|---|---:|
| Training | 2016 |
| Validation | 504 |
| Standard Test | 372 |
| Custom Images | 10 |

---

## Data Preprocessing

Data preprocessing is an important part of the project because the CNN needs all input images to have a consistent format.

The preprocessing pipeline uses `torchvision.transforms`.

The main steps are:

```text
Original Image
      ↓
Resize
      ↓
Convert to Tensor
      ↓
Normalize
      ↓
CNN Model
