# Approval System Analysis

This project investigates algorithmic fairness in automated approval systems by comparing three decision-making approaches:

- **Behavior-Based Model**: Approves based on financial and behavioral indicators.
- **Biased Model**: Introduces social bias (e.g., gender, marital status).
- **Fairer Model**: Removes bias to provide a more equitable outcome.

---

## 🎯 Objective

To show how biases (like gender) can be introduced into logical systems and how these affect approval decisions across different groups.

---

## 📂 Dataset

Fields include:
- `AMT_INCOME_TOTAL`
- `NAME_INCOME_TYPE`
- `overdue_months`
- `CODE_GENDER`
- `NAME_FAMILY_STATUS`

---

## ⚙️ Methods

### 🔹 Rule-Based Approvals
Three custom functions determine approvals:
- `behavior_based_approval(row)`
- `biased_approval(row)`
- `fairer_model(row)`

### 🔹 Evaluation
- Used `classification_report` for precision, recall, F1-score.
- Evaluated subgroup fairness by gender.

### 🔹 SHAP Explanation
- Trained `XGBoostClassifier` on biased data.
- Visualized feature impact using SHAP beeswarm plot.

---

## 📊 Approval Rates by Gender

| Gender | Biased Approval | Behavior-Based | Fairer Model |
|--------|------------------|----------------|--------------|
| F      | 0.000000         | 0.242793       | 0.242793     |
| M      | 0.359597         | 0.412123       | 0.412123     |

**Interpretation**: The biased model severely disadvantages women even with comparable income and credit records.

---

## 🧪 SHAP Feature Importance

The SHAP plot revealed:
- `CODE_GENDER` and `NAME_FAMILY_STATUS` had significant influence in the **biased model**, proving discriminatory behavior.

---

## 🛠️ Tools

- Python, Pandas, NumPy
- Scikit-learn, SHAP, XGBoost
- Jupyter Notebook

---

## 📁 Project Structure

├── data/
├── notebooks/
│ └── 04_fairness_metrics_analysis.ipynb
├── fairness_dashboard.py
├── report.md
├── README.md


---

## 👤 Author

**Harish Kummara**  
MSc Data Science for Society and Business  
Constructor University  
[GitHub](https://github.com/vadaayar)

---

## 📚 Course Reference

- **Course**: Introduction to Computational Social Science  
- **Professor**: Jan Lorenz  
- [Course GitHub](https://github.com/janlorenz/CU-S25_MDSSB-SOCB-02-Introduction-Computational-Social-Science)
