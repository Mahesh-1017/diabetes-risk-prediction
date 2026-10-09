import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes.csv")

cols = ["Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"]


for col in cols:
    df[col] = df[col].replace(0, np.nan)

print(df.isnull().sum())
cols = ["Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, np.nan)

print(df.isnull().sum())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25,
    random_state=42, stratify=y_temp
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    )
}

trained_models = {}

for name, model in models.items():

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    trained_models[name] = pipe

    print(name, "trained successfully")



from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

results = []

for name, model in trained_models.items():

    predictions = model.predict(X_val)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print(results_df.round(3))


results_df.set_index("Model").plot(
    kind="bar", figsize=(10, 5)
)

plt.title("Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

best_model_name = results_df.sort_values(
    ["Recall", "Precision"], ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("Selected model:", best_model_name)


probabilities = best_model.predict_proba(X_test)[:, 1]
limit = int(np.ceil(len(X_test) * 0.30))


selected_indices = np.argsort(probabilities)[-limit:]

final_predictions = np.zeros(len(X_test), dtype=int)
final_predictions[selected_indices] = 1

print("Patients in test set:", len(X_test))
print("Patients selected for testing:", limit)

print("\nFinal evaluation:")
print(classification_report(
    y_test, final_predictions, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_predictions))


cm = confusion_matrix(y_test, final_predictions)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Not Diabetic", "Diabetic"],
    yticklabels=["Not Diabetic", "Diabetic"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Diabetes Prediction Confusion Matrix")
plt.show()