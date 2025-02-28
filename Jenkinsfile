pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = 'http://localhost:5000'  // MLflow server URL
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                echo 'Building the project...'
                // Run the model pipeline script that includes MLflow logging
                sh 'python model_pipeline.py'  // Ensure model_pipeline.py contains MLflow logging
            }
        }

        stage('Test') {
            steps {
                echo 'Running tests...'
                // Add your test scripts here if you have any
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying the application...'
                // Deployment commands go here
            }
        }

        stage('Log with MLflow') {
            steps {
                script {
                    // Log the model with MLflow (if not done in model_pipeline.py)
                    sh 'python -m mlflow.log_model model_pipeline.py --experiment-name="Experiment1"'
                }
            }
        }
    }
}
