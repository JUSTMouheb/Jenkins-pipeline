import sys
import os

# Modify sys.path to include the src directory at the very top
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# Now import the model_pipeline after sys.path modification
from model_pipeline import load_model  # noqa: E402


def test_model_loading():
    """Test if the model loads correctly."""
    model = load_model("model.pkl")
    assert model is not None, "Model should not be None"  # nosec B101


# Run the test if executed directly
if __name__ == "__main__":
    test_model_loading()
