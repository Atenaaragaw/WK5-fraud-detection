# WK5-Fraud-Detection: E-Commerce & Banking Risk Analysis

## 📌 1. Project Overview & Business Objective
For **Adey Innovations Inc.**, fraud detection is a critical pillar of financial stability. This project implements a machine learning pipeline to:
* **Maximize Revenue Protection:** Identifying fraudulent transactions before processing.
* **Optimize Customer Experience:** Minimizing "False Positives" that frustrate legitimate users.
* **Botnet Detection:** Utilizing behavioral velocity features to identify automated attacks.

---

## 📂 2. Repository Structure
- `data/`: Segregated into `raw/` and `processed/` (engineered features).
- `models/`: Contains the serialized `random_forest_fraud_model.pkl`.
- `notebooks/`: Comprehensive `modeling.ipynb` containing the full experimental pipeline.
- `src/`: Production-ready Python scripts for data preprocessing.

---

## 🛠️ 3. Task 1: Data Engineering & Robustness
* **Geolocation Mapping:** Implemented a robust `merge_asof` lookup to map IP addresses to countries.
* **Feature Engineering:** Developed "Velocity Features" (e.g., `time_since_signup`, `device_freq`) to capture bot-like behavior.
* **Robustness:** Integrated **Try-Except error handling** in the data pipeline to ensure graceful failures during ingestion.
* **Imbalance Handling:** Applied **SMOTE** strictly to training data to prevent data leakage while addressing class imbalance.

---

## 🤖 4. Task 2: Model Building & Methodological Rigor
To ensure rigorous results, we compared our advanced models against a statistical baseline and optimized them via automated tuning.

### Performance Comparison:
| Metric | Logistic Regression (Baseline) | Random Forest (Tuned Winner) |
| :--- | :--- | :--- |
| **AUC-PR Score** | 0.0910 | **0.6870** |
| **Recall (Class 1)**| 0.00 | **0.58** |
| **Precision** | 0.00 | **0.74** |

### Key Improvements & Methodology:
* **Baseline Comparison:** The **Logistic Regression** baseline failed to identify any fraud (Recall 0.0), proving that linear models cannot capture the complex, non-linear patterns in this dataset.
* **Hyperparameter Tuning:** Conducted a **GridSearchCV** with **3-Fold Stratified Cross-Validation**.
* **Optimal Parameters:** `$max\_depth: 20, n\_estimators: 200, min\_samples\_split: 2$`.
* **Business Result:** The tuned Random Forest provides a **74% Precision** rate, significantly reducing the cost of false alarms for the business.



---

## 🚀 5. Roadmap: Task 3 & 4
### Task 3: Model Explainability (In Progress)
* **Objective:** Use **SHAP** (SHapley Additive exPlanations) to provide "Reason Codes" for flagged transactions.
* **Current Status:** Generating summary plots to identify the top behavioral triggers for fraud.



### Task 4: Deployment & Monitoring
* **Strategy:** Implementing a Flask API for real-time inference and monitoring for **Model Drift** over time.

---

## 🧪 6. How to Run
1. **Activate Environment:** `.\venv\Scripts\Activate.ps1`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Pipeline:** Execute the `notebooks/modeling.ipynb` for the full training suite.