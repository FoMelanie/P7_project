# app.py
from typing import Optional
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# create FastAPI app and load model
app = FastAPI(title = 'Home Credit Prediction - P7 project',
              description='XGBoost model is used for prediction')
model = joblib.load('xgb_weight_pipeline_custom_scorer.joblib')

class Data(BaseModel):
    customer_id: int

@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {'message': 'System is healthy'}

# load data with features
all_data = pd.read_csv('cleaned_data_most_important_features.csv')
all_data = all_data.set_index('SK_ID_CURR')

# define model prediction function using the loaded model and data with features
def predict_from_data(customer_id:Data,client_data:pd.DataFrame,model):
    """Predict the client probability to repay its loan from the bank according to the selected model and its features.

    Args:
        client_id (int): the client id number from which we want to determine the probability to repay the loan
        client_data (pd.DataFrame): pandas dataframe containing clients data (features) used for prediction. Each row corresponds to one client
        model: the model used to predict the probability

    Returns:
        client_prediction: float corresponding to the client probability to repay its loan
    """
    # Get client features values and reshape into numpy array
    customer_values_list = client_data.loc[int(customer_id)].values.tolist()
    customer_values_np_array = np.array(customer_values_list)
    customer_values_reshape = customer_values_np_array.reshape(1, -1)
    
    # Get model prediction
    customer_prediction = model.predict_proba(customer_values_reshape)
    
    return customer_prediction[0][1].item()

# Create an endpoint that receives GET requests and returns prediction value of target class = 1
@app.get('/predict/{customer_id}')
def predict(customer_id):
    predictions = predict_from_data(customer_id, all_data, model)
    return {'Prediction': predictions}