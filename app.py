from flask import Flask,render_template,json,jsonify,request
import joblib
import pickle
import numpy as np
from pathlib import Path
import os
import pandas as pd


app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[float(x) for x in request.form.values()]
    print(features)
    columns = ['area1', 'area2', 'area3', 'compactness1', 'compactness2', 'compactness3', 'concave_points1', 'concave_points2', 'concave_points3', 'concavity1', 'concavity2', 'concavity3', 'fractal_dimension1', 'fractal_dimension2', 'fractal_dimension3', 'perimeter1', 'perimeter2', 'perimeter3', 'radius1', 'radius2', 'radius3', 'smoothness1', 'smoothness2', 'smoothness3', 'symmetry1', 'symmetry2', 'symmetry3', 'texture1', 'texture2', 'texture3']
    df = pd.DataFrame({columns[i]: [value] for i, value in enumerate(features)})
    print(df.head())
    
    # data pre-processing
    columns_lst_with_zero_std_dev = []
    columns_with_zero_std_dev_path = Path("artifacts/preprocessed_data/columns_with_zero_std_dev.pkl")
    if os.path.exists(columns_with_zero_std_dev_path):
        with open(columns_with_zero_std_dev_path, "rb") as pkl_file:
            columns_lst_with_zero_std_dev = pickle.load(pkl_file)

    if len(columns_lst_with_zero_std_dev) > 0:
        df = df.drop(columns=columns_lst_with_zero_std_dev, axis=1)
    
    # standard scaling
    std_scaler_path = Path("artifacts/preprocessed_data/std_scaler.pkl")
    with open(std_scaler_path, "rb") as file:
        std_scaler = pickle.load(file)

    feature_names = std_scaler.feature_names_in_
    print(feature_names)
    df_scaled = pd.DataFrame(std_scaler.transform(df), columns=df.columns)
    
    # load the final model
    final_model_path = Path("artifacts/model_training/final_model/final_model.joblib")
    final_model = joblib.load(final_model_path)
    feature_names = final_model.feature_names_in_
    print(feature_names)
    prediction=final_model.predict(df_scaled.values)
    print(prediction)

    prediction = 1
    if prediction == 1:
        return render_template('results.html', prediction="Malignant")
    elif prediction == 0:
        return render_template('results.html', prediction="Benign")


if __name__ == '__main__':
    app.run(debug=True)

