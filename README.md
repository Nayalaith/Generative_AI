# Sudoku Classification Project

## Introduction

This project aims to explore the learning capabilities of neural network-based models, specifically in understanding complex positional relationships. While CNNs can analyze individual images, we want to test whether a neural network can infer relationships when images are presented together, as in a Sudoku puzzle. Sudoku requires the understanding that each number must appear only once in every row and column. Instead of relying on separate algorithms, we aim to determine if the neural network can learn this concept on its own.

---

## Dataset
The dataset for this project is available on Kaggle: [Sudoku Images Based on MNIST](https://www.kaggle.com/datasets/laithnayal/sudoku-images-based-on-mnist). The final files used:

- `sudoku_training_set.h5` (400k images)
- `sudoku_testing_set.h5` (30k images)
- `sudoku_validation_set.h5` (30k images)

### Dataset Details
The images were generated using the code provided in the `Dataset` directory of the associated repository. The dataset was created using MNIST digits, carefully partitioned into training, validation, and testing sets to ensure that the validation and test sets contain unseen data. The images adhere to Sudoku rules, ensuring unique digits in each row, column, and 3x3 grid. Each image is unique, with no repeated samples.

---

## Experiments and Evaluation

This project explored two neural network architectures to classify Sudoku and non-Sudoku images using a dataset of 84x84 pixel images. Below are the details of each architecture:


### A. Variational Auto-Encoder (VAE)
The VAE builds on the AE by introducing probabilistic components, enabling better generalization through a distributional latent space.

#### Architecture:
- **Encoder**: Outputs the mean and log-variance of the latent space for reparameterization.
- **Decoder**: Reconstructs the input from the sampled latent vector.
- **Classifier**: A fully connected layer applied on the latent vector.

---

### B. Convolutional Neural Network (CNN)
The CNN architecture uses convolutional layers to extract hierarchical spatial features, with fully connected layers for classification.

#### Architecture:
- **Convolutional Layers**: Four layers with ReLU activations and max-pooling.
- **Fully Connected Layers**: Used after feature extraction for classification.
- **Output Layer**: Sigmoid activation function for binary classification.

---

## Breaking Through Performance Plateaus

### Challenges:
Initially, all models performed poorly:
- **CNN Accuracy**: 50-65%
- **VAE Accuracy**: Even lower than CNN.

### Solutions:
1. **Dataset Expansion**: Increased the dataset size from 100k to 400k training samples, ensuring balanced classes and greater diversity.
2. **Hyperparameter Tuning**: Extensively tuned hyperparameters to improve model performance.
3. **Model Refinement**: Enhanced the CNN architecture to better capture intricate patterns.

These changes helped the CNN achieve a breakthrough in performance, reaching a testing accuracy of **87%**, while AE and VAE showed limited improvements.

---

## Final Results

| Model                  | Accuracy |
|------------------------|----------|
| Variational Auto-Encoder (VAE) | 68.56%   |
| Convolutional Neural Network (CNN) | **87.00%**   |

---

## Usage

### 1. Use the Pre-trained Models
You can find the trained models at the following drive link: [Trained Models](https://drive.google.com/drive/folders/1YmfZWEdJoFGLlJcq2NteSS9qLh5Nd2tN?usp=sharing).

For testing:

A separate testing notebook was created to evaluate the performance of the two models. This notebook is designed to:

1. Load the testing dataset.
2. Load the trained models: Variational Auto-Encoder (VAE), and Convolutional Neural Network (CNN).
3. Run the testing dataset through each model to obtain predictions.
4. Present a selection of test examples along with insights into the model performances.

You can use  **testing_models.ipynb** located in the **Testing Notebooks** directory of this repository.

---

### 2. Train Your Own Models

To generate your own dataset, you can use the notebook **Sudoku_Dataset_Generation.ipynb**, which is designed to create the Sudoku and non-sudoku dataset based on MNIST digits.

For training your own models, the following notebooks are available in the **code** directory:

- **CNN.ipynb**: This notebook contains the code to train the Convolutional Neural Network (CNN).
- **variational-autoencoder.ipynb**: This notebook contains the code to train the Variational Auto-Encoder (VAE).


## Conclusion
This project demonstrated the efficiancy of CNNs in classifying Sudoku and non-Sudoku images, particularly when combined with a larger and more diverse dataset. Despite initial struggles, the CNN significantly outperformed AE and VAE architectures, establishing its robustness and capacity for generalization.
