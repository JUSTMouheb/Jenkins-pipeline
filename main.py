import joblib
import argparse
import mlflow
import mlflow.sklearn
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ML Pipeline Execution")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    print(
        f"Arguments: prepare={args.prepare}, train={args.train}, evaluate={args.evaluate}"
    )

    # Define file paths
    train_file = "churn8.csv"
    test_file = "churn2.csv"

    # Prepare data (if --prepare is specified)
    if args.prepare:
        print("Preparing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = prepare_data(
            train_file, test_file
        )

        # Save the label encoders
        joblib.dump(label_encoders, "label_encoders.pkl")
        print("Label encoders saved to 'label_encoders.pkl'.")
        print("Data preparation completed.")
        return  # Exit after preparing data

    # Prepare data (if --train or --evaluate is specified)
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoders = prepare_data(
        train_file, test_file
    )

    # Set MLflow experiment
    mlflow.set_experiment("Churn Prediction Experiment")

    # Train the model
    if args.train:
        print("Training the model...")
        with mlflow.start_run():
            # Hyperparameters for training (modify based on your model)
            n_estimators = 100  # Example hyperparameter
            max_depth = 10  # Example hyperparameter

            # Log hyperparameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Train the XGBoost model
            model = train_model(X_train, y_train)

            # Save the trained model
            save_model(model, "model.pkl")

            # Log the model with MLflow
            mlflow.sklearn.log_model(model, "xgboost_model")

            # Log evaluation metrics (accuracy, etc.)
            accuracy = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

        print("Training and logging completed.")

    # Evaluate the model
    if args.evaluate:
        print("Evaluating the model...")
        model = load_model("model.pkl")
        with mlflow.start_run():
            accuracy = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
        print(f"Evaluation completed with accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
