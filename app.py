from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            satisfaction_level=float(request.form.get('satisfaction_level')),
            last_evaluation=float(request.form.get('last_evaluation')),
            number_project=int(request.form.get('number_project')),
            average_montly_hours=int(request.form.get('average_montly_hours')),
            time_spend_company=int(request.form.get('time_spend_company')),
            work_accident=int(request.form.get('Work_accident')),
            promotion_last_5years=int(request.form.get('promotion_last_5years')),
            departments=request.form.get('Departments '),
            salary=request.form.get('salary')

        )
       
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)[0]
        print(results)
        if results==0.0:
            print(results)
            return render_template('home.html',prediction_text="Employee won't leave the company")
        else:
            return render_template('home.html',prediction_text="Employee will leave the company")
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True,port=5000)        