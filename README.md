# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN) 🖼️📊

This project implements a Convolutional Neural Network (CNN) model to classify images from the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 classes. The network architecture involves convolutional layers for feature extraction and fully connected layers for classification, optimized using both **Adam** and **SGD** optimizers to compare performance. Activation functions like **Leaky ReLU** and **Sigmoid** are used, and the model's generalization is further improved with **Dropout** layers.

## Dataset 🗂️

The **CIFAR-10** dataset contains 10 different image classes, including vehicles, animals, and other objects. Below is a sample images of the dataset:  
![image](https://github.com/user-attachments/assets/198a1b16-52dc-4c34-b9ab-543af08bba20)


## Model Architecture 🧠

The model is built using a typical **Convolutional Neural Network (CNN)** architecture, as depicted below. The architecture includes:
- **Convolutional Layers** for feature extraction.
- **Pooling Layers** to reduce dimensionality.
- **Fully Connected Layers** for classification.  
![image](https://github.com/user-attachments/assets/7146b1b3-43a9-4637-9bef-c9e632bd6273)


## Key Features 📌

- **Optimizer Comparison**: The performance of the **Adam** and **SGD** optimizers is evaluated to determine which achieves better training accuracy.
- **Activation Functions**: **Leaky ReLU** and **Sigmoid** activations are compared in the model's performance.
- **Dropout Regularization**: Dropout is applied to improve generalization and reduce overfitting.
  
## Results 📈

The model achieves high accuracy after training for 50 epochs. Below is the comparative analysis of training accuracy using different activation functions and optimizers.

## How to Run 🛠️

To run the model, open **`CIFAR10_DeepLearning_Classification.ipynb`** in Jupyter Notebook or Google Colab and execute the cells.
