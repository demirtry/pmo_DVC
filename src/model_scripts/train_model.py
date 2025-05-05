import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import joblib


def train(config):
    df_train = pd.read_csv(config['data_split']['train_path'])
    df_test  = pd.read_csv(config['data_split']['test_path'])

    X_train,y_train = df_train.drop(columns = ["Pass_Fail"]).values, df_train["Pass_Fail"].values
    X_test, y_test = df_test.drop(columns = ["Pass_Fail"]).values, df_test["Pass_Fail"].values

    param_grid = {
        'C': config['train']['C'],
        'penalty': config['train']['penalty'],
        'solver': config['train']['solver'],
        'max_iter': config['train']['max_iter'],
    }

    mlflow.set_experiment("my experiment")

    with mlflow.start_run():
        lr = LogisticRegression(max_iter=10000)
        grid_search = GridSearchCV(lr, param_grid, cv=3, n_jobs=4)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        for param, value in best_params.items():
            mlflow.log_param(param, value)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"acc: {accuracy}")
        print(f"f1_score: {f1}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if label != 'accuracy':
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, "my_model", signature=signature)

        with open(config['train']['model_path'], "wb") as file:
            joblib.dump(best_model, file)

        dfruns = mlflow.search_runs()
        path2model = dfruns.sort_values("metrics.f1_score", ascending=False).iloc[0]['artifact_uri'].replace("file://", "") + '/my_model'
        print(path2model)
