# titanic-survival-prediction-ml
 Machine Learning classification model predicting passenger survival on the Titanic using Scikit-Learn and Kaggle dataset.
# 🚢 Titanic Survival Prediction — Machine Learning Project

An end-to-end Machine Learning classification project based on the classic **Kaggle Titanic: Machine Learning from Disaster** competition. This project predicts passenger survival rates by analyzing demographic, socio-economic, and family structure attributes.

---

## 📌 Problem Statement & Objectives

The goal is to build a predictive binary classification model that answers the question: *"What factors made passengers more likely to survive the Titanic disaster?"*

Key objectives:
1. Perform Exploratory Data Analysis (EDA) to identify survival correlations across age, gender, class, and family size.
2. Feature engineer raw inputs (extracting titles from passenger names, grouping family size, imputing missing values).
3. Train and compare multiple classification algorithms to maximize predictive accuracy and ROC-AUC score.

---

## 📊 Dataset Overview

Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)

| Feature | Description | Type |
| :--- | :--- | :--- |
| **`Survived`** | Target Variable (0 = No, 1 = Yes) | Binary Categorical |
| **`Pclass`** | Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd) | Ordinal Categorical |
| **`Sex`** | Gender (male / female) | Categorical |
| **`Age`** | Age in years | Numerical (Continuous) |
| **`SibSp`** | # of siblings / spouses aboard | Numerical (Discrete) |
| **`Parch`** | # of parents / children aboard | Numerical (Discrete) |
| **`Fare`** | Passenger Fare | Numerical (Continuous) |
| **`Embarked`** | Port of Embarkation (C, Q, S) | Categorical |

---

## 🛠️ Data Preprocessing & Feature Engineering

1. **Missing Value Imputation:**
   * Median age imputation based on `Pclass` and `Sex` groupings.
   * Mode imputation for missing `Embarked` values.
2. **Feature Extraction:**
   * **`Title`**: Extracted titles (`Mr`, `Mrs`, `Miss`, `Master`, `Officer`) from the `Name` column.
   * **`FamilySize`**: Combined `SibSp` + `Parch` + 1 to calculate total group size.
   * **`IsAlone`**: Binary indicator showing whether a passenger traveled solo.
3. **Encoding & Scaling:**
   * One-Hot Encoding applied to `Sex`, `Embarked`, and `Title`.
   * Standard scaling applied to continuous features (`Age`, `Fare`).

---

## 🤖 Models & Performance Evaluation

Multiple classification models were trained and evaluated using **5-Fold Cross-Validation**:

| Algorithm | Training Accuracy | Test Accuracy | ROC-AUC Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~79.5% | ~78.2% | 0.82 |
| **Decision Tree Classifier** | ~82.1% | ~79.1% | 0.80 |
| **Random Forest Classifier** | ~85.4% | ~82.6% | 0.86 |
| **Gradient Boosting (XGBoost)**| **~86.2%** | **~83.8%** | **0.88** |

*The Random Forest and Gradient Boosting models demonstrated the highest predictive performance after hyperparameter tuning.*

---

## 💡 Key Business & Domain Insights

* **Gender Bias:** Female passengers had a significantly higher survival rate (~74%) compared to male passengers (~19%), reflecting the "women and children first" protocol.
* **Socio-Economic Class:** 1st Class passengers had a ~63% survival rate compared to only ~24% for 3rd Class passengers.
* **Family Size Impact:** Passengers traveling in small families (2–4 members) had higher survival rates than solo travelers or large families (>4 members).

---

## 🧰 Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Processing:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Model Selection, Preprocessing, Ensemble Methods)
* **Environment:** Jupyter Notebook / Google Colab / VS Code
