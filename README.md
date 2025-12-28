# WK5-Fraud-Detection: E-Commerce & Banking Risk Analysis

## 📌 1. Project Overview & Business Objective
For **Adey Innovations Inc.**, fraud detection is a critical pillar of financial stability and customer trust. The objective of this project is to build an automated machine learning system that:
* **Maximizes Revenue Protection:** Identifying fraudulent transactions before they lead to chargebacks.
* **Optimizes Customer Experience:** Ensuring high precision to avoid "False Positives" that block legitimate users.
* **Botnet Detection:** Utilizing behavioral velocity features to identify automated attacks.

---

## 📂 2. Repository Structure
* `.github/workflows/`: CI/CD pipelines for automated unit testing.
- `data/`: Contains `raw/` datasets and `processed/` files ready for modeling.
- `models/`: Saved `.pkl` files of the best-performing models.
- `notebooks/`: Exploratory Data Analysis (EDA) and Model Training experiments.
- `src/`: Production-ready Python scripts for preprocessing.
- `tests/`: Unit tests to ensure data integrity.

---

## 🛠️ 3. Task 1: Data Engineering (Corrected)
* **Geolocation Mapping:** Implemented a robust `merge_asof` lookup to map IP addresses to countries accurately.
* **Feature Engineering:** Developed "Velocity Features" including `time_since_signup`, `user_freq`, and `device_freq` to capture behavioral patterns.
* **Handling Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) strictly to the training set to prevent data leakage while providing the model enough fraud samples to learn from.



---

## 🤖 4. Task 2: Model Building & Evaluation
We compared two ensemble architectures to determine the best fit for the business objective.

### Performance Comparison:
| Metric | Random Forest (Winner) | XGBoost |
| :--- | :--- | :--- |
| **AUC-PR Score** | **0.6983** | 0.6899 |
| **Recall (Fraud)** | **0.62** | 0.58 |
| **Precision (Fraud)** | 0.64 | **0.78** |
| **F1-Score** | 0.63 | 0.66 |

### Business Decision:
The **Random Forest** model was selected for the final pipeline. While XGBoost has higher precision, the Random Forest’s superior **Recall (0.62)** ensures we catch 4% more fraudulent transactions, which is prioritized for initial risk mitigation.



---

## 🚀 5. Roadmap & Future Tasks

### Task 3: Model Explainability (In Progress)
* **Objective:** Use **SHAP** (SHapley Additive exPlanations) to explain individual predictions.
* **Technique:** TreeExplainer for Random Forest.
* **Goal:** Provide "Reason Codes" for the fraud team (e.g., "Flagged due to high device frequency and low time-to-purchase").

### Task 4: Monitoring & Deployment
* **Challenge:** Fraudster behavior shifts over time (Model Drift).
* **Mitigation:** Establishing a monitoring loop to track performance degradation and trigger automated retraining.

---

## 🧪 6. How to Run
1. **Activate Environment:** `.\venv\Scripts\Activate.ps1`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Pipeline:** Execute the `notebooks/modeling.ipynb` or `src/preprocessing.py`.