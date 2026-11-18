Project 2: Supervised Learning – Fraud Detection Pipeline
DecodeLabs Internship – Data Science Track

Description
This project focuses on building a robust fraud detection system using supervised learning on a highly imbalanced credit card transaction dataset. The main goals are:

- Handling extreme class imbalance using SMOTE
- Training classification models (Logistic Regression and Random Forest)
- Evaluating models using Precision, Recall, and ROC-AUC (instead of Accuracy)


Dataset
-Source: Credit Card Fraud Detection Dataset (ULB)
-Total Transactions: 284,807
-Fraud Cases: 492 (0.173%)
-Features: Time, V1–V28 (PCA transformed), Amount, Class


## How to Run

### Requirements
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn

Steps
1.Download the dataset (creditcard.csv) and place it in the same folder as the notebook.
2. Open the Jupyter Notebook.
3. Run all cells from top to bottom in order.
4. The final selected model is Logistic Regression.

Project Workflow

1. Exploratory Data Analysis
Confirmed extreme class imbalance (99.83% legitimate vs 0.17% fraud)

2. Train-Test Split
Performed an 80/20 stratified split to maintain the same class ratio in both sets

3. Handling Imbalance (SMOTE)
Applied SMOTE only on the training set (leak-free approach)
Balanced the training data to 227,451 samples for each class

4. Model Training
Logistic Regression (with feature scaling)
Random Forest Classifier

5. Model Evaluation
Focused on Precision, Recall, and ROC-AUC
Accuracy was deliberately ignored as required by the project

Key Results
| Metric                    | Score      |
|---------------------------|------------|
| Precision (Fraud)         | 0.13       |
| Recall (Fraud)            | 0.90       |
| F1-Score (Fraud)          | 0.23       |
| ROC-AUC                   | 0.9768     |
| False Negatives           | 10         |
| False Positives           | 571        |

Key Decisions
Applied SMOTE only on the training data to avoid data leakage
Selected Logistic Regression as the final model because it achieved the highest ROC-AUC and the best Recall
Prioritized high Recall to minimize missed fraud cases