# TODO: add necessary import
import pytest
import pickle
import numpy as np
import pandas as pd
from ml.model import train_model, compute_model_metrics, load_model, inference
from ml.data import process_data


# TODO: implement the first test. Change the function name and input as needed
# first test tests the functionality of training model 
def test_train_model():
    # generate synthetic data for testing
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(2, size=100)
    
    # training the model
    model = train_model(X_train, y_train)
    
    # check if the model is trained
    assert model is not None


# TODO: implement the second test. Change the function name and input as needed
# second test tests the compute model metrics functions and checks if metrics are populated and within range
def test_compute_model_metrics():
    # generate synthetic data for testing
    y_true = np.random.randint(2, size=100)
    y_pred = np.random.randint(2, size=100)
    
    # compute model metrics
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # check if metrics are generated and within range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

# TODO: implement the third test. Change the function name and input as needed
# third test tests the inference function and checks if predictions are generated
def test_inference():
    # loading our model
    model_path = "model/model.pkl"
    model = load_model(model_path)
    
    # generate synthetic data for testing
    X_test = np.random.rand(10, 108)
    
    # running the dummy data against the our model and receiving predictions on the 10 dummy features
    predictions = inference(model, X_test)
    
    # check if predictions are generated
    assert len(predictions) == 10