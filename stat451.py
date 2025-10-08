# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the dataset
path = "distinct_crops.csv"
# path = "Crop_recommendation.csv"
data = pd.read_csv(path)

print(data.head())
print(data.describe())

print(data['label'].value_counts()) # 10 labels

print(data[data['N'] == 120])
# ---- 1. Preprocessing ----

# Identify features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Encode the target variable (e.g., crop labels)
le = LabelEncoder()
y = le.fit_transform(y)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0, stratify=y)

# ---- 2. Models ----

# Define individual models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "kNN": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(random_state=0, max_iter=500),
    "SVM": SVC(kernel='linear', random_state=0)
}

# Ensemble model (Voting Classifier)
ensemble_model = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier(random_state=0)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('lr', LogisticRegression(random_state=0, max_iter=500))
], voting='hard')

# models["Ensemble"] = ensemble_model

# ---- 3. Evaluation ----

# Prepare to store evaluation metrics
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": []
}

# K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Train and evaluate each model
for name, model in models.items():
    # Cross-validate
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='precision_macro')
    print(f"{name} - precision after CV: {np.mean(cv_scores):.3f}")

    # Train the model on the full training set
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')

    # Store the results
    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["Precision"].append(prec)
    results["Recall"].append(rec)

    # Print the classification report
    print(f"--- {name} Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---- 4. Results ----

# Convert results to a DataFrame for easy comparison
results_df = pd.DataFrame(results)

# Sort models by Accuracy
results_df = results_df.sort_values(by="Precision", ascending=False)

# Display the results
print("\nModel Comparison:")
print(results_df)


import matplotlib.pyplot as plt
import seaborn as sns

# Plot confusion matrix for each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


# Save results to a CSV file
# results_df.to_csv("model_comparison_results.csv", index=False)
