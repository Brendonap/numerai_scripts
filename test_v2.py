import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

def main():
    # Set seed for reproducibility
    np.random.seed(0)

    #training_data_set = 'Round110/numerai_training_data.csv'
    training_data_set = 'Round110/numerai_training_data.csv'
    prediction_data_set = 'Round110/numerai_tournament_data.csv'

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv(training_data_set, header=0)
    prediction_data = pd.read_csv(prediction_data_set, header=0)

    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]

    x_prediction = prediction_data[prediction_data["data_type"].isin(["validation", "live", "test"])]
    x_prediction_features = x_prediction[features]
    ids = x_prediction["id"]

    clf = XGBRegressor(objective="reg:logistic")
    parameters = {'n_estimators': [50,75,100], 'max_depth':[10,15,20]}
    model = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, n_jobs=4, scoring='neg_mean_squared_log_error', verbose=True)

    print("Training...")
    model.fit(X, Y)

    print("Predicting...")
    y_prediction = model.predict(x_prediction_features)

    print("Writing predictions to predictions.csv")
    results = pd.DataFrame(data={'id': ids, 'probability': y_prediction})
    results.to_csv("predictions.csv", index=False)

    if (True):
        validation_data = x_prediction[x_prediction["data_type"] == "validation"]
        validation_id_target = validation_data[["id", "target"]]
        
        # or reslts.id in (select id from validation_target) 
        validation_prediction = results[results["id"].isin(validation_id_target["id"])] 

        log_loss_result = metrics.log_loss(validation_id_target["target"], validation_prediction['probability'])
        print("log loss result")
        print(log_loss_result)

if __name__ == '__main__':

    main()