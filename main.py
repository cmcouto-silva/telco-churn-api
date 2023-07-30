import pickle
import uvicorn
import pandas as pd
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

# Instantiate API
app = FastAPI()

# Load models
with open('models/cluster_pipeline.pkl', 'rb') as file:
    cluster_pipeline = pickle.load(file)

with open('models/prediction_pipeline.pkl', 'rb') as file:
    prediction_pipeline = pickle.load(file)

# List input model features
FEATURES = [
    'CLTV', 'Contract', 'Dependents', 'Device Protection', 'Internet Service',
    'Monthly Charges', 'Multiple Lines', 'Online Backup', 'Online Security',
    'Paperless Billing', 'Partner', 'Payment Method', 'Senior Citizen',
    'Streaming Movies', 'Streaming TV', 'Tech Support', 
    'Tenure Months', 'Total Charges'
    ]

input_feature_names = [
    'cltv', 'contract', 'dependents', 'device_protection', 'internet_service',
    'monthly_charges', 'multiple_lines', 'online_backup', 'online_security',
    'paperless_billing', 'partner', 'payment_method', 'senior_citizen',
    'streaming_movies', 'streaming_tv', 'tech_support',
    'tenure_months', 'total_charges'
    ]

feature_input_mapper = {input: feature for input,feature in zip(input_feature_names, FEATURES)}

# Homepage
@app.get('/')
def home():
    return 'Welcome to the Churn Prediction API!'

# What if classification
@app.get('/predict')
def predict(
    cltv: int=4400,
    contract: str='Month-to-month', 
    dependents: str='No',
    device_protection: str='No',
    internet_service: str='Fiber optic',
    monthly_charges: float=65,
    multiple_lines: str='No',
    online_backup: str='No',
    online_security: str='No',
    paperless_billing: str='Yes',
    partner: str='No',
    payment_method: str='Electronic check',
    senior_citizen: str='No',
    streaming_movies: str='Yes',
    streaming_tv: str='Yes',
    tech_support: str='No',
    tenure_months: int=32,
    total_charges: float=2283,
    ):

    # Input data
    df_input = pd.DataFrame([{
        'CLTV': cltv,
        'Contract': contract,
        'Dependents':dependents,
        'Device Protection':device_protection,
        'Internet Service': internet_service,
        'Monthly Charges': monthly_charges,
        'Multiple Lines': multiple_lines,
        'Online Backup': online_backup,
        'Online Security': online_security,
        'Paperless Billing': paperless_billing,
        'Partner': partner,
        'Payment Method': payment_method,
        'Senior Citizen': senior_citizen,
        'Streaming Movies': streaming_movies,
        'Streaming TV': streaming_tv,
        'Tech Support': tech_support,
        'Tenure Months': tenure_months,
        'Total Charges': total_charges
        }])
    
    # Cluster prediction
    try:
        df_input['cluster'] = cluster_pipeline.predict(df_input)
    except:
        df_input['cluster'] = -1

    # Churn prediction
    output = prediction_pipeline.predict(df_input)[0]

    return int(output)



class Customer(BaseModel):

    cltv: int
    contract: str
    dependents: str
    device_protection: str
    internet_service: str
    monthly_charges: float
    multiple_lines: str
    online_backup: str
    online_security: str
    paperless_billing: str
    partner: str
    payment_method: str
    senior_citizen: str
    streaming_movies: str
    streaming_tv: str
    tech_support: str
    tenure_months: int
    total_charges: float

    class Config:
        schema_extra = {
            'example': {
                'cltv': 4400,
                'contract': 'Month-to-month', 
                'dependents': 'No',
                'device_protection': 'No',
                'internet_service': 'Fiber optic',
                'monthly_charges': 65,
                'multiple_lines': 'No',
                'online_backup': 'No',
                'online_security': 'No',
                'paperless_billing': 'Yes',
                'partner': 'No',
                'payment_method': 'Electronic check',
                'senior_citizen': 'No',
                'streaming_movies': 'Yes',
                'streaming_tv': 'Yes',
                'tech_support': 'No',
                'tenure_months': 32,
                'total_charges': 2283
            }
        }

@app.post('/predict_with_json')
def predict(data: Customer):
    df_input = pd.DataFrame([data.dict()]).rename(columns=feature_input_mapper)

    # Cluster prediction
    try:
        df_input['cluster'] = cluster_pipeline.predict(df_input)
    except:
        df_input['cluster'] = -1

    # Churn prediction
    output = prediction_pipeline.predict(df_input)[0]

    return int(output)


class CustomerList(BaseModel):
    data: List[Customer]

@app.post('/mult_predict_with_json')
def predict(data: CustomerList):
    df_input = pd.DataFrame(data.dict()['data']).rename(columns=feature_input_mapper)

    # Cluster prediction
    try:
        df_input['cluster'] = cluster_pipeline.predict(df_input)
    except:
        df_input['cluster'] = -1

    # Churn prediction
    output = prediction_pipeline.predict(df_input).tolist()

    return output

# Executa API
if __name__ == '__main__':
    uvicorn.run(app)
