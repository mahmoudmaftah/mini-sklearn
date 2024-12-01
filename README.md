
# Mini Sklearn

**Mini Sklearn** is a simplified implementation of the popular Python machine learning library, Scikit-learn. This project focuses on creating key components of Scikit-learn from scratch, offering a deep understanding of how machine learning algorithms and tools work under the hood. It serves as an educational resource for developers and students who want to explore the inner workings of machine learning libraries.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Benchmarks](#benchmarks)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## About the Project

This project aims to replicate the core functionality of Scikit-learn by implementing commonly used machine learning algorithms, preprocessing techniques, and evaluation metrics. Mini Sklearn demonstrates the foundational concepts of machine learning in a modular, extensible architecture.

**Key Highlights:**
- Implemented multiple models and learning algorithms such as Linear Regression, Naive Bayes, Decision Trees, and Neural Networks.
- Designed modules for data preprocessing, imputation, ensembling, and model selection.
- Benchmarked performance against Scikit-learn’s equivalent models.

---

## Features

- **Supervised Learning Models**: Includes Linear Regression, Naive Bayes, Neural Networks, and Decision Trees.
- **Preprocessing Techniques**: Handles missing data, scaling, encoding, and normalization.
- **Ensemble Methods**: Basic implementations of bagging and boosting techniques.
- **Custom Evaluation Metrics**: Reimplementation of scoring metrics like accuracy, precision, and recall.
- **Benchmarking**: Comparative performance analysis with Scikit-learn implementations.

---

## Folder Structure

```plaintext
mini-sklearn/
│
├── _helpers/           # Utility functions and reusable components
├── ensemble/           # Implementation of ensemble methods
├── impute/             # Data imputation strategies
├── linear_model/       # Linear regression and related models
├── metrics/            # Custom evaluation metrics
├── model_selection/    # Model selection utilities (e.g., train-test split)
├── naive_bayes/        # Naive Bayes classifier
├── neighbors/          # K-Nearest Neighbors implementation
├── neural_network/     # Basic neural network implementation
├── preprocessing/      # Data preprocessing techniques
├── testing/            # Unit tests for implemented models
├── tree/               # Decision tree implementations
│
├── BaseClasses.py      # Abstract base classes for consistency
├── __init__.py         # Package initializer
│
└── benchmarks/         # Scripts for benchmarking against Scikit-learn
```

---

## Benchmarks

A key component of this project is comparing the performance of the implemented algorithms with Scikit-learn. Benchmarks include:
- **Speed**: Measuring training and prediction time.
- **Accuracy**: Evaluating results on various datasets.
- **Scalability**: Testing performance with larger datasets.

Detailed results are available in the `benchmarks/` folder.

---

## Getting Started

### Prerequisites

Ensure you have Python installed along with the necessary packages. You can install the dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/mini-sklearn.git
   cd mini-sklearn
   ```

2. **Run tests**:
   ```bash
   python -m unittest discover testing/
   ```

---

## Usage

Below is an example of how to use Mini Sklearn for linear regression:

```python
from linear_model import LinearRegression

# Sample dataset
X = [[1], [2], [3]]
y = [2, 4, 6]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict([[4]])
print(predictions)  # Output: [8]
```

---

## Future Enhancements

- Adding support for unsupervised learning algorithms such as K-Means and PCA.
- Extending support for more ensemble methods like Random Forest.
- Improved documentation and visualization tools.
- Packaging the library for distribution via PyPI.

---

## Contributing

Contributions are welcome! If you have suggestions or find any issues, please create a pull request or open an issue on GitHub.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
