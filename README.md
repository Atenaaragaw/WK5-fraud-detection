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

## 🚀 5. Roadmap: Task 3 
### Task 3: Model Explainability (In Progress)
* **Objective:** Use **SHAP** (SHapley Additive exPlanations) to provide "Reason Codes" for flagged transactions.
* **Current Status:** Generating summary plots to identify the top behavioral triggers for fraud.

---

## 🧪 6. How to Run
1. **Activate Environment:** `.\venv\Scripts\Activate.ps1`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Pipeline:** Execute the `notebooks/modeling.ipynb` for the full training suite.

## 🏛️ Executive Summary: Fraud Detection System
Objective: To develop a high-precision machine learning pipeline that identifies fraudulent e-commerce transactions while minimizing friction for legitimate customers.

## 1. The Challenge
The initial data revealed a massive class imbalance (approx. 90% legitimate vs. 10% fraud). Standard statistical models (Logistic Regression) failed completely, yielding 0% recall on fraud. This necessitated a sophisticated, non-linear approach to protect revenue.

## 2. The Solution: "Advanced Ensemble Intelligence"
We implemented a Random Forest Classifier optimized through GridSearchCV and Stratified Cross-Validation.

Precision: 0.74 — When the system flags fraud, it is correct 74% of the time.

AUC-PR: 0.6870 — A massive improvement over the 0.091 baseline, showing superior ability to distinguish risk in imbalanced environments.

Robustness: Integrated SMOTE to balance training data and comprehensive Try-Except blocks for data ingestion reliability.

## 3. Key Behavioral Drivers (The "Why")
Using SHAP Explainability, we identified the specific behaviors that trigger fraud alerts:

Device Velocity (device_freq): The single strongest predictor. Fraud is highly correlated with "device-sharing," where one device is used to access multiple accounts in a short window.

Signup Maturity (time_since_signup): High-risk transactions occur almost immediately after account creation.

Channel Risk (source): Specific marketing channels show a significantly higher density of fraudulent activity.

## 4. Strategic Recommendations
To immediately reduce fraud losses at Adey Innovations, we recommend:

Velocity Throttling: Implement a hard block or 24-hour hold on any device associated with more than two unique User IDs.

Friction-Based Authentication: Trigger mandatory Multi-Factor Authentication (MFA) for users with less than 30 minutes of "account age" if their purchase value exceeds a specific threshold.

Source-Based Auditing: Re-evaluate marketing spend on the "sources" identified by SHAP as high-risk to eliminate bot-driven referral traffic.