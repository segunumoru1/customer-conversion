import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, accuracy_score
import mlflow
from mlflow import sklearn
from mlflow.models.signature import infer_signature
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    # Ensure artifacts directory exists
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Load data
    df = pd.read_csv("C:\\Users\\SEGUN\\customer-conversion\\data\\week2_marketing_data.csv")
    if "converted" not in df.columns:
        raise ValueError("Target column 'converted' not found in the dataset.")
    df.dropna(inplace=True)
    if "customer_id" in df.columns:
        df.drop("customer_id", axis=1, inplace=True)

    # Plot class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='converted', data=df)
    plt.title("Distribution of 'converted' Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "converted_class_distribution.png"))
    plt.close()

    X = df.drop("converted", axis=1)
    y = df["converted"]

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=42)
    }

    best_acc = 0
    best_model = None
    best_model_name = ""
    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", float(acc))
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                mlflow.log_metric("roc_auc", float(roc_auc))
            else:
                y_pred_proba = None
                roc_auc = None

            # Add signature and input_example
            signature = infer_signature(X_test, y_pred)
            input_example = X_test[:1]
            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=input_example
            )

            # Save confusion matrix plot for each model
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(artifacts_dir, f"confusion_matrix_{name}.png"))
            plt.close()

            results[name] = {
                "model": model,
                "accuracy": acc,
                "roc_auc": roc_auc,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba
            }
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_model_name = name

    print(f"Best model: {best_model_name} with accuracy {best_acc:.4f}")

    # Use the best model's predictions for reporting
    y_pred = results[best_model_name]["y_pred"]
    y_pred_proba = results[best_model_name]["y_pred_proba"]

    print(f"Classification Report for {best_model_name}:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print("ROC AUC Score:", roc_auc)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, "roc_curve.png"))
        plt.close()

    # Feature importance for Random Forest
    rf_model = results["RandomForest"]["model"]
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=X.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    print("Feature Importances:\n", feature_importances)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
    plt.title('Feature Importances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "feature_importances.png"))
    plt.close()

    # k-fold cross-validation
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = scores
        print(f"{name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Log the best model with MLflow
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.log_metric("best_model_accuracy", float(best_acc))
    mlflow.end_run()

    # Save artifacts
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(artifacts_dir, "model_results.csv"))
    feature_importances.to_csv(os.path.join(artifacts_dir, "feature_importances.csv"))
    joblib.dump(results, os.path.join(artifacts_dir, "results.pkl"))
    joblib.dump(best_model, os.path.join(artifacts_dir, "customer_conversion_model.pkl"))

# Run the training function
if __name__ == "__main__":
    train_model()