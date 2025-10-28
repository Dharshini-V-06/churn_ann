#  Customer Churn Prediction using Artificial Neural Network (ANN)

This project builds a deep learning model to predict whether a bank customer is likely to **churn (exit)** or **stay**, based on demographic and financial features.
It uses the **Churn_Modelling.csv** dataset and implements an **ANN** using **TensorFlow/Keras**.

---

##  Dataset Overview

**Dataset link:** [Churn Dataset](https://www.kaggle.com/code/shrutimechlearn/deep-tutorial-1-ann-and-classification/input)


**Dataset:** `Churn_Modelling.csv`
**Rows:** 10,000
**Columns:** 14

| Feature         | Description                                              |
| --------------- | -------------------------------------------------------- |
| CreditScore     | Customer's credit score                                  |
| Geography       | Country of residence (France, Spain, Germany)            |
| Gender          | Male/Female                                              |
| Age             | Customer's age                                           |
| Tenure          | Number of years the customer has been with the bank      |
| Balance         | Account balance                                          |
| NumOfProducts   | Number of products the customer uses                     |
| HasCrCard       | Whether the customer has a credit card (1 = Yes, 0 = No) |
| IsActiveMember  | Whether the customer is active (1 = Yes, 0 = No)         |
| EstimatedSalary | Estimated annual salary                                  |
| Exited          | Target variable (1 = Exited, 0 = Stayed)                 |

---

##  Project Workflow

###  Data Preprocessing

* Loaded dataset using `pandas`
* Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
* Checked for:

  * Missing values → **None found**
  * Duplicates → **None found**
* Detected and capped **outliers** using the **IQR method**
* Applied **log transformation** to reduce skewness in `Age`
* Encoded categorical variables:

  * `Gender` → Label Encoding
  * `Geography` → One-hot Encoding
* Standardized features using **StandardScaler**

---

###  Exploratory Data Analysis (EDA)

* **Boxplots** to visualize outliers for numerical features
* **Correlation Heatmap** to observe relationships between variables
* Checked class imbalance in target variable (`Exited`):

  * 0 → 7,963 customers stayed
  * 1 → 2,037 customers exited

---

###  Model Architecture

| Layer | Units | Activation | Description                          |
| ----- | ----- | ---------- | ------------------------------------ |
| Dense | 32    | ReLU       | Input + Hidden Layer 1               |
| Dense | 16    | ReLU       | Hidden Layer 2                       |
| Dense | 2     | Softmax    | Output Layer (binary classification) |

---

###  Model Compilation & Training

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metrics:** Accuracy
* **Epochs:** 50
* **Batch Size:** 8

---

###  Model Evaluation

| Metric   | Training Set | Testing Set |
| -------- | ------------ | ----------- |
| Accuracy | **86.09%**   | **82.80%**  |

The model generalizes well with minimal overfitting.

---



##  Technologies Used

* **Python 3.x**
* **NumPy**
* **Pandas**
* **Matplotlib & Seaborn**
* **Scikit-learn**
* **TensorFlow / Keras**

---

##  Key Insights

* Age and CreditScore have strong influence on churn probability.
* Active members are less likely to churn.
* Customers with multiple products tend to stay longer.

---

##  Future Improvements

* Apply **SMOTE** to handle class imbalance
* Add **Dropout layers** to reduce overfitting
* Perform **Hyperparameter Tuning** (batch size, learning rate, epochs)
* Try other models (Random Forest, XGBoost) for comparison

---

##  Results Summary

 Model trained successfully
 No missing values or duplicates
 Achieved **82.8% accuracy on test data**
 Clean, reproducible workflow

---

