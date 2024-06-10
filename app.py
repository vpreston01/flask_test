import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
import sqlite3
from datetime import datetime


churro = "sqlite:///mpgdb.db"
# engine = create_engine(churro)


# churro = "postgresql://postgres:postgresql@104.155.61.55/postgres"
engine = create_engine(churro)



with open("model.pkl", "rb") as f:
    saved_model = pickle.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def welcome():
    return "<h1>MPG MODEL PREDICTOR"

@app.route("/predict", methods=["GET"])
def predict():

    cylinders = request.args.get("cylinders", None)
    displacement = request.args.get("displacement", None)
    horsepower = request.args.get("horsepower", None)
    weight = request.args.get("weight", None)
    acceleration = request.args.get("acceleration", None)
    model_year = request.args.get("model_year", None)
    origin = request.args.get("origin", None)

    data = [cylinders, displacement, horsepower, weight,
            acceleration, model_year, origin]
    
    if None in data:
        return str(-999)
    else:
        pred_df = pd.DataFrame(np.array(data).reshape(1,-1), 
                               columns=saved_model.feature_names_in_)
        
        inputs = str(data)
        outputs = str(saved_model.predict(pred_df)[0])
        date = str(datetime.now())[0:19]
        log_df = pd.DataFrame({"inputs":[inputs], 
                               "outputs": [outputs], 
                               "date": [date]})
        log_df.to_sql("logs", con=engine, if_exists="append", index=None)

        return outputs
    
@app.route("/check_logs", methods=["GET"])
def check_logs():
    return pd.read_sql("SELECT * FROM logs", con=engine).to_html()




if __name__ == "__main__":
    app.run(debug=True)

