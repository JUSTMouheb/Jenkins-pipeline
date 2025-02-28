# Define Python and Virtual Environment
PYTHON=python3
VENV=venv

# Default target
all: install prepare train evaluate test ci security

# Install dependencies
install:
	@echo "Setting up virtual environment and installing dependencies..."
	@$(PYTHON) -m venv $(VENV)
	@. $(VENV)/bin/activate && pip install -r requirements.txt

# Prepare data
prepare:
	@echo "Preparing data..."
	@. $(VENV)/bin/activate && python main.py --prepare

# Run training
train:
	@echo "Training the model..."
	@. $(VENV)/bin/activate && python main.py --train

# Run evaluation
evaluate:
	@echo "Evaluating the model..."
	@. $(VENV)/bin/activate && python main.py --evaluate

# Run tests
test:
	@echo "Running tests..."
	@. $(VENV)/bin/activate && pytest tests/

# CI/CD steps
ci: format lint security

# Format code using black
format:
	@echo "Formatting code with black..."
	@. $(VENV)/bin/activate && black .

# Lint code using flake8
lint:
	@echo "Linting code with flake8..."
	@. $(VENV)/bin/activate && flake8 .

# Run security checks with bandit
security:
	@echo "Running security checks with bandit..."
	@. $(VENV)/bin/activate && bandit -r . -x ./venv

# Run the FastAPI app and open Swagger UI
run-api:
	@echo "Starting FastAPI app and opening Swagger UI..."
	@. $(VENV)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Clean cache and old models
clean:
	@echo "Cleaning up unnecessary files..."
	@rm -rf __pycache__ *.pkl *.log $(VENV)
