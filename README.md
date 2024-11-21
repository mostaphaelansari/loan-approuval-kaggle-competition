# Loan Approval Prediction using PyCaret

This project focuses on predicting loan approval statuses using a machine learning pipeline developed with PyCaret. It was part of a Kaggle Playground competition and achieved a **90% accuracy**, showcasing effective use of synthetic datasets and automated machine learning.

---

## Project Overview

Loan approval prediction is a crucial problem in financial services. This project leverages advanced data preprocessing, exploratory data analysis (EDA), and automated model selection through PyCaret to address the challenge.

- **Tools Used**: PyCaret, Pandas, Matplotlib, Seaborn
- **Dataset**: Synthetic dataset provided by Kaggle for Playground competitions.

---

## Approach

### 1. **Data Preprocessing**
- Handled missing values and dropped irrelevant columns like `id`.
- Converted categorical features (`loan_intent`, `loan_grade`) into dummy variables.
- Applied `astype('category')` for categorical columns like `person_home_ownership`.
- Normalized numerical columns using PyCaret's built-in functionalities.

### 2. **Exploratory Data Analysis (EDA)**
- Analyzed target variable distribution (`loan_status`) with pie and bar plots.
- Investigated key features like `loan_intent` for insights into data composition.

### 3. **Modeling with PyCaret**
- Set up the PyCaret classification environment with:
  - Target: `loan_status`
  - Normalization: Enabled
  - Categorical and numerical feature specifications.
- Evaluated models using PyCaret's `compare_models()` to identify the best-performing algorithms.

### 4. **Results**
- **Top Models**:
  - **Light Gradient Boosting Machine (LGBM)**: Achieved the highest accuracy of 95.17%.
  - **XGBoost**: Close second with an accuracy of 95.09%.
  - Other models evaluated include Random Forest, Gradient Boosting Classifier, and Extra Trees Classifier.
- **Final Accuracy**: 90% on the test dataset.

---


---

## Key Learnings

- PyCaret accelerates the machine learning pipeline, allowing efficient experimentation.
- Synthetic datasets from Kaggle Playground competitions provide a great balance between realism and privacy.

---

## Usage

### Prerequisites
Install the required libraries:
```bash
pip install pycaret pandas matplotlib seaborn
