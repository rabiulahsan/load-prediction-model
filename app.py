from flask import Flask,request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd 
import math
import os
from src.exception import CustomException
from src.logger import logging
from src.pipelines.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application
CORS(app)

# Transform 'age' into categories
def age_category(age):
    if age <= 25:
        return "Young"
    elif 26 <= age <= 50:
        return "Middle aged"
    else:
        return "Senior"

# Route for a home page
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Loan Eligibility Prediction API! Using GitHub Action as a CiCd tool with Render."})

# Route for making predictions
@app.route('/predictdata', methods=['POST'])
def predictdata():
    try:
        #log the route hit
        print("hitting the route successfully...")  
        # Retrieve data from the request JSON
        data = request.json

        age_as_string = age_category(int(data.get('age')))
        
        
        # Create a CustomData instance with the input data
        custom_data = CustomData(
            age=age_as_string,
            income_stability=data.get('income_stability'),
            co_applicant=int(data.get('co_applicant')),
            income=int(data.get('income')),
            current_loan=int(data.get('current_loan')),
            credit_score=int(data.get('credit_score')),
            loan_amount_request=int(data.get('loan_amount_request')),
            property_price=int(data.get('property_price'))
        )

        
        # Convert the input data to a DataFrame
        pred_df = custom_data.make_data_frame()
        
        # Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredictPipeline()
        classification_result = predict_pipeline.predict_approval(pred_df)

        # Return the result as a JSON response
        if(classification_result[0] == 1):
            #getting predicted loan amount
            loan_amount_result = predict_pipeline.predict_loan_amount(pred_df)
            #getting requested loan amount
            loan_amount_req =  int(data.get('loan_amount_request'))
            print(loan_amount_req)
                
            loan_amount = (math.ceil(round(loan_amount_result[0]) / 100)) * 100
            print(loan_amount)
            if (loan_amount < loan_amount_req):
                return jsonify({"loan_amount": loan_amount})
            else:
                return jsonify({"loan_amount": loan_amount_req})

        else:
            return jsonify({"loan_amount": 0})
        

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)





    # body should like this 

#     {
#   "age": 65,
#   "income_stability": "Low",
#   "co_applicant": 1,
#   "income": 1664.05,
#   "current_loan": 400,
#   "credit_score": 781,
#   "loan_amount_request": 72448,
#   "property_price": 113464
# }
