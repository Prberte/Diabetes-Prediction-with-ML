# Diabetes-Prediction-with-ML

This repository contains a machine learning project focused on predicting diabetes using various classification algorithms. The project demonstrates the application of machine learning techniques to a medical dataset and explores multiple approaches to improve prediction accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Diabetes is a chronic disease that affects millions of people worldwide. Early detection and management are crucial to improving patient outcomes. This project uses machine learning models to predict whether a patient has diabetes based on a set of medical attributes.

## Dataset

The dataset used for this project is the Pima Indians Diabetes Database, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes). The dataset consists of several medical predictor variables (independent variables) and one target variable indicating the presence or absence of diabetes.

### Features

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history)
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1), 0 indicates non-diabetic and 1 indicates diabetic

## Project Structure

The repository is organized as follows:

```plaintext
.
├── data
│   └── diabetes.csv          # Dataset used for the project
├── notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis notebook
│   ├── Model_Training.ipynb  # Model training and evaluation notebook
│   └── Model_Comparison.ipynb # Model comparison and final selection
├── src
│   ├── data_preprocessing.py # Data preprocessing scripts
│   ├── model.py              # Model training scripts
│   └── evaluation.py         # Model evaluation scripts
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Requirements

To run this project, you will need Python 3.x and the packages listed in `requirements.txt`. You can install these packages using `pip`.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Prberte/Diabetes-Prediction-with-ML.git
    cd Diabetes-Prediction-with-ML
    ```

2. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use the Jupyter notebooks provided in the `notebooks` directory to understand the workflow of the project. 

- **EDA.ipynb**: Run this notebook to explore the dataset, visualize distributions, and understand the relationships between features.
- **Model_Training.ipynb**: Use this notebook to train various machine learning models such as Logistic Regression, Decision Trees, Random Forests, and others.
- **Model_Comparison.ipynb**: This notebook compares the performance of the trained models and selects the best-performing one.

## Modeling and Evaluation

This project employs various machine learning models, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation and grid search are used to optimize the models.

## Results

The best model achieved an accuracy of **X%** on the test set. Detailed evaluation results, including confusion matrices and ROC curves, can be found in the `Model_Comparison.ipynb` notebook.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README to better suit the specific details and results of your project.
