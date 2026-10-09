# 🩺 Diabetes Risk Prediction Using Machine Learning

A beginner-friendly machine learning project that predicts diabetes risk using the **Pima Indians Diabetes Database** and compares three classification algorithms.

The project explores how machine learning can help a rural health clinic prioritize patients for further testing when laboratory capacity is limited to **30% of daily walk-ins**.

> **Note:** This project is for educational purposes only. It is not a medical diagnostic tool and should not be used to make real patient-care decisions.

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Project Workflow](#-project-workflow)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Evaluation Metrics](#-evaluation-metrics)
- [Business Objective](#-business-objective)
- [Project Structure](#-project-structure)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

## 🎯 About the Project

Diabetes is a serious health condition, and identifying people who may be at risk can help healthcare professionals decide who needs further examination.

In this project, I use Python and machine learning to predict whether a patient may have diabetes based on medical and demographic features.

I train and compare three machine learning models:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier

The project focuses not only on prediction accuracy but also on recall, precision, and the clinic's limited testing capacity.

## 🏥 Problem Statement

Imagine a rural health clinic where many patients visit every day, but the clinic can perform a full blood test on only 30% of them.

Testing patients randomly may result in some high-risk patients being missed.

The goal of this project is to:

1. Estimate each patient's diabetes risk.
2. Rank patients according to their predicted risk.
3. Prioritize the highest-risk 30% for further testing.
4. Evaluate how many actual diabetic patients are identified and how many non-diabetic patients are unnecessarily selected.

The model supports patient prioritization in this simulated scenario. A real clinic would need clinical validation before using such a system.

## 📊 Dataset

**Dataset:** Pima Indians Diabetes Database

**Source:** [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/code/mragpavank/pima-indians-diabetesdatabase/input)

The dataset contains medical measurements and a target column indicating diabetes status.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family-history-related score |
| Age | Age in years |
| Outcome | Target: 1 = diabetic, 0 = non-diabetic |

Some medical columns contain zero values that may represent missing measurements. These values are handled during preprocessing, while valid zeros such as zero pregnancies are preserved.

## 🤖 Machine Learning Models

### 1. K-Nearest Neighbors (KNN)

KNN predicts a patient's class by comparing the patient with similar examples in the training data. Feature scaling is important for this algorithm.

### 2. Logistic Regression

Logistic Regression estimates the probability of diabetes and provides a useful baseline for binary classification.

### 3. Decision Tree Classifier

A Decision Tree learns decision rules from the input features and can capture non-linear relationships.

The models are compared using the same training and validation data to identify a suitable candidate for the clinic's testing-capacity constraint.

## 🔄 Project Workflow

1. Load the CSV dataset using Pandas.
2. Inspect the data and identify missing or invalid measurements.
3. Replace invalid zero measurements with missing values where appropriate.
4. Split the data into training, validation, and testing sets using stratification.
5. Fill missing values using the median and scale features where needed.
6. Train KNN, Logistic Regression, and Decision Tree models.
7. Compare accuracy, precision, recall, and F1-score.
8. Rank patients by predicted diabetes probability.
9. Evaluate the selected model under the 30% testing limit.
10. Review the confusion matrix and final test results.

## 🛠️ Technologies Used

- **Python** — Main programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Confusion matrix visualization
- **Scikit-learn** — Preprocessing, model training, and evaluation
- **VS Code** — Development environment
- **Git and GitHub** — Version control and project hosting

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later recommended
- Visual Studio Code
- Git
- The dataset CSV file

### 1. Clone the repository

```bash
git clone https://github.com/Mahesh-1017/diabetes-risk-prediction.git
cd diabetes-risk-prediction
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Download the dataset

Download the CSV from the [Kaggle dataset page](https://www.kaggle.com/code/mragpavank/pima-indians-diabetesdatabase/input).

Place the CSV in the project folder and make sure its filename matches the filename used in the Python program.

For example:

```text
diabetes.csv
```

### 5. Run the project

```bash
python main.py
```

The program should display model evaluation results in the terminal and generate the relevant plots.

## 📈 Evaluation Metrics

The project uses the following metrics:

| Metric | Why it matters |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures how many selected patients actually have diabetes |
| Recall | Measures how many diabetic patients are identified |
| F1-score | Balances precision and recall |
| Confusion Matrix | Shows true positives, false positives, true negatives, and false negatives |

For this problem, recall is important because missing a diabetic patient can be costly. Precision is also important because the clinic has limited testing resources.

## 🎯 Business Objective: 30% Testing Capacity

The clinic can test only 30% of its daily walk-ins.

Instead of relying only on a default probability threshold of 0.5, the project ranks patients by predicted diabetes probability and selects the highest-risk group within the testing limit.

The model should be assessed on:

- The number and proportion of actual diabetic patients identified.
- The number of non-diabetic patients selected for testing.
- Recall and precision among the selected patients.
- Whether the 30% capacity limit is respected.

Model selection should be performed using validation data, with final evaluation on previously unseen test data.

## 📁 Project Structure

```text
diabetes-risk-prediction/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── diabetes.csv
```

The dataset file is optional if users download it separately from Kaggle.

## ⚠️ Limitations

- The dataset represents a specific population and may not generalize to other populations.
- The dataset is relatively small compared with many modern machine learning datasets.
- Some zero measurements may represent missing data rather than genuine values.
- A high predicted risk does not confirm that a patient has diabetes.
- Model performance must be checked on unseen data and validated before any real-world use.
- The 30% testing constraint is a simulated business requirement, not a validated clinical policy.

## 🔮 Future Improvements

- Add exploratory data analysis and feature visualizations.
- Compare additional algorithms such as Random Forest and Support Vector Machine.
- Tune model parameters using validation data.
- Compare models under the same 30% testing-capacity constraint.
- Add precision-recall curves and threshold analysis.
- Build a simple Streamlit interface for demonstration.
- Evaluate fairness and generalization on representative external datasets.

## 👨‍💻 Author

**Mahesh Vinnakota**

B.Tech — Artificial Intelligence and Machine Learning

- GitHub: [Mahesh-1017](https://github.com/Mahesh-1017)
- LinkedIn: [Mahesh Vinnakota](https://www.linkedin.com/in/mahesh-vinnakota)

---

⭐ If you find this project useful for learning machine learning, feel free to star the repository.

*Built as an academic machine learning project to explore classification, model evaluation, and healthcare resource prioritization.*