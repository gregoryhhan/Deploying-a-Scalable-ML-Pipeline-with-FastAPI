# TODO: add necessary import
import pytest
import numpy as np
from ml.model import train_model, compute_model_metrics, inference, save_model, load_model, performance_on_categorical_slice
from ml.data import process_data

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    # Generate synthetic data for testing
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(2, size=100)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Check if the model is trained (not None)
    assert model is not None


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    # Generate synthetic data for testing
    y_true = np.random.randint(2, size=100)
    y_pred = np.random.randint(2, size=100)
    
    # Compute model metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Check if the metrics are within valid range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

# TODO: implement the third test. Change the function name and input as needed
def test_inference():
    # Load a trained model (assuming a trained model is available)
    model_path = "model/model.pkl"
    model = load_model(model_path)
    
    # Generate synthetic data for testing inference
    X_test = np.random.rand(10, 108)
    
    # Perform inference
    predictions = inference(model, X_test)
    
    # Check if predictions are generated
    assert len(predictions) == 10
