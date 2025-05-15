# Breast-Cancer-Prediction

**Breast-Cancer-Prediction** is a machine learning project designed to classify breast cancer tumors as benign or malignant based on diagnostic features. This project uses Python and popular ML libraries to train and evaluate models on the Breast Cancer Wisconsin dataset.

## 🚀 Features

- Loads and processes the Breast Cancer dataset.
- Trains machine learning models (e.g., Logistic Regression, SVM, Decision Tree).
- Evaluates performance using metrics such as accuracy, precision, recall, and F1-score.
- Supports model comparison and visualization via confusion matrix or graphs.

## 🛠️ Prerequisites

- **Python 3.x**
- Required Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

## 📦 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/Vinayakrai/Breast-Cancer-Prediction.git
cd Breast-Cancer-Prediction
```

2. **Set Up Environment and Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, manually install:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 Usage

1. Open the Jupyter notebook:

```bash
jupyter notebook breast_cancer_prediction.ipynb
```

2. Run the notebook step by step:
   - Load dataset
   - Preprocess data
   - Train and evaluate models
   - Visualize results

3. Optionally, run a standalone script if provided:

```bash
python predict.py
```

## 📁 Project Structure

```
Breast-Cancer-Prediction/
├── breast_cancer_prediction.ipynb  # Main notebook
├── predict.py                      # Optional script
├── dataset.csv                     # Dataset (if included)
├── requirements.txt                # Python dependencies
└── README.md
```

