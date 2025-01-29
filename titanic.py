import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("train.csv")

# Feature Engineering
df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())  # Extract title from name
df["FamilySize"] = df["SibSp"] + df["Parch"]  # Family size feature
df["IsAlone"] = (df["FamilySize"] == 0).astype(int)  # Alone indicator
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Encode gender

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"]
X = df[features]
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

first_class_idx = X_train["Pclass"] == 1
expert_first_class = RandomForestClassifier(n_estimators=100, random_state=42)
expert_first_class.fit(X_train[first_class_idx], y_train[first_class_idx])

male_idx = X_train["Sex"] == 0
expert_gender = RandomForestClassifier(n_estimators=100, random_state=42)
expert_gender.fit(X_train[male_idx], y_train[male_idx])

family_idx = X_train["FamilySize"] > 0
expert_family = GradientBoostingClassifier(n_estimators=100, random_state=42)
expert_family.fit(X_train[family_idx], y_train[family_idx])

# 4️⃣ Children Expert (age-based expert)
child_idx = X_train["Age"] < 10
expert_child = GradientBoostingClassifier(n_estimators=100, random_state=42)
expert_child.fit(X_train[child_idx], y_train[child_idx])

gating_network = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

gating_features = ["Pclass", "Sex", "FamilySize"]
X_gating = X_train[gating_features]

# Define labels based on survival tendencies
y_gating = np.where(X_train["Pclass"] == 1, 0,  # First-Class Expert
             np.where(X_train["Sex"] == 0, 1,  # Gender Expert
                      np.where(X_train["FamilySize"] > 0, 2,  # Family Expert
                               3)))  # Children Expert

gating_network.fit(X_gating, y_gating)

# ========================== Predict using Mixture of Experts ==========================

def mixture_of_experts_predict(X):
    """Predict survival using Mixture of Experts"""
    X_gating_input = X[["Pclass", "Sex", "FamilySize"]]  # Use the actual feature names
    expert_choice = gating_network.predict(X_gating_input)  # Decide which expert to trust

    pred_first = expert_first_class.predict_proba(X)[:, 1]  # 1st Class Expert
    pred_gender = expert_gender.predict_proba(X)[:, 1]  # Gender Expert
    pred_family = expert_family.predict_proba(X)[:, 1]  # Family Expert
    pred_child = expert_child.predict_proba(X)[:, 1]  # Children Expert

    expert_probs = np.vstack([pred_first, pred_gender, pred_family, pred_child]).T
    expert_weights = gating_network.predict_proba(X_gating_input)  # Get confidence scores
    final_pred = np.sum(expert_probs * expert_weights, axis=1)  # Weighted sum

    return (final_pred > 0.5).astype(int)

# ========================== Evaluate Model ==========================

y_pred = mixture_of_experts_predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Mixture of Experts Accuracy: {accuracy:.4f}")

# ========================== Visualize the Results ==========================
survival_counts = pd.Series(y_pred).value_counts()
labels = ['Survived', 'Not Survived']
plt.bar(labels, survival_counts, color=['green', 'red'])
plt.title('Survival Prediction Distribution')
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.show()