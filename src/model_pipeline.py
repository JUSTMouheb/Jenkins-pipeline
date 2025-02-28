import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import mlflow


def prepare_data(train_file, test_file):
    """Prepares data for model training and testing."""
    # Load datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Print column names to check the correct target column
    print("Training data columns:", train_data.columns)
    print("Test data columns:", test_data.columns)

    # Split the data into features and target variable
    X = train_data.drop("Churn", axis=1)  # Ensure 'Churn' matches the exact column name
    y = train_data["Churn"]  # Ensure 'Churn' matches the exact column name

    # Using train_test_split to split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test = test_data.drop(
        "Churn", axis=1
    )  # Ensure 'Churn' matches the exact column name
    y_test = test_data["Churn"]  # Ensure 'Churn' matches the exact column name

    # Label encoding (if necessary)
    label_encoders = {}
    for column in X_train.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])
        X_val[column] = le.transform(X_val[column])
        X_test[column] = le.transform(X_test[column])
        label_encoders[column] = le

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoders


def train_model(X_train, y_train):
    """Trains the XGBoost model."""
    # Define the model
    model = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy: {:.2f}%".format(accuracy * 100))

    # Log accuracy to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Additional metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Log classification report to MLflow
    mlflow.log_metric(
        "precision", class_report["accuracy"]
    )  # Overall accuracy as example
    mlflow.log_metric("recall", class_report["macro avg"]["recall"])
    mlflow.log_metric("f1_score", class_report["macro avg"]["f1-score"])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Optionally, save the confusion matrix plot and log it as artifact
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    return accuracy


def plot_feature_importance(model, X_train):
    """Plots the feature importance of the trained model."""
    feature_importance = model.feature_importances_
    features = X_train.columns

    # Sort feature importance
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importance[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()


def load_model(model_file):
    """Loads a saved model from a file."""
    return joblib.load(model_file)


def save_model(model, model_file):
    """Saves the trained model to a file."""
    joblib.dump(model, model_file)


# Assuming you're calling the following to start a run in MLflow
if __name__ == "__main__":
    # Example usage:
    with mlflow.start_run():
        # Prepare data
        train_file = "train_data.csv"
        test_file = "test_data.csv"
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = prepare_data(
            train_file, test_file
        )

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save the model
        save_model(model, "xgb_model.pkl")
