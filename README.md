# WK5-Fraud-Detection: E-commerce & Banking

## 📌 Project Overview
This project aims to improve fraud detection for **Adey Innovations Inc.** by building robust machine learning models that identify fraudulent transactions in e-commerce and credit card data. The project involves geolocation integration, behavioral feature engineering, and handling extreme class imbalance.

## I. Business Objective (The "Why")
For Adey Innovations Inc., fraud is not just a technical error; it is a direct drain on the bottom line.Financial Protection: Every missed fraud case (False Negative) results in direct revenue loss and chargeback fees.Customer Trust: Every blocked legitimate customer (False Positive) causes friction, potentially leading to churn.Objective: Build a precision-targeted system that maximizes fraud detection (Recall) while minimizing customer annoyance (Precision).
## II. Roadmap for Tasks 2 & 3
TaskObjectiveModels/MetricsChallengesTask 2: ModelingIdentify best-performing algorithm.Random Forest, XGBoost. Metrics: AUC-PR, F1-Score.Overfitting: Synthetic SMOTE data can bias the model. Mitigation: Stratified Cross-Validation.Task 3: ExplainabilityTranslate "Black Box" into "Human Logic."SHAP, LIME values.Complexity: Non-linear models are hard to explain. Mitigation: Use SHAP summary plots.



## 📂 Repository Structure
- `.github/workflows/`: CI/CD pipelines (GitHub Actions).
- `data/`: Raw and processed datasets (Note: Raw data is gitignored).
- `notebooks/`: Jupyter notebooks for EDA and Model training.
- `src/`: Production-ready Python scripts for preprocessing and modeling.
- `tests/`: Unit tests for the data pipeline.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```powershell
   git clone [https://github.com/Atenaaragaw/WK5-fraud-detection.git](https://github.com/YOUR_USERNAME/WK5-fraud-detection.git)
   cd WK5-fraud-detection
2. **Create and Activate Virtual Environment:**
python -m venv venv
.\venv\Scripts\Activate.ps1
3. **Install Dependencies:**
pip install -r requirements.txt
**🚀 Completed Tasks**
Task 1: Data Analysis and Preprocessing
Geolocation Integration: Mapped IP addresses to countries using range-based lookups.

Feature Engineering: Developed velocity features like time_since_signup and device_freq.

EDA: Conducted univariate and bivariate analysis to identify fraud patterns by geography and behavior.

Class Imbalance: Applied SMOTE to balance the training set, moving from a 9% minority class to a 50/50 balanced distribution.

**📊 Key Insights from EDA**
Velocity Matters: Users who purchase immediately after signing up have a significantly higher probability of being flagged as fraud.

Geographic Hotspots: Certain countries exhibit higher fraud rates, which are now captured in our categorical features.

Class Imbalance: The dataset was highly imbalanced, requiring synthetic oversampling (SMOTE) for the models to learn effectively.

**🧪 Testing**
Automated tests are handled via GitHub Actions. To run tests locally:
python -m unittest discover tests
🛠️ Technologies Used
Python 3.13

Pandas & NumPy: Data manipulation.

Scikit-Learn: Preprocessing and Scaling.

Imbalanced-Learn: SMOTE implementation.

Matplotlib & Seaborn: Visualization.
