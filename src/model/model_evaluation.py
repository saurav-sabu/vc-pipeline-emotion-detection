import pandas as pd
import numpy as np

import os
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

def load_model(model_path):
    clf = pickle.load(open(model_path,"rb"))
    return clf

def load_data(test_data_path):
    test_data = pd.read_csv(test_data_path)
    return test_data

def split_X_test_y_test(test_data):
    X_test = test_data.iloc[:,:-1].values
    y_test = test_data.iloc[:,-1].values
    return X_test,y_test

def prediction(X_test,y_test,clf):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred,y_pred_proba

def evaluate_model(y_pred,y_test,y_pred_proba):
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metric_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

    return metric_dict

def save_metrics(metric_dict):
    json.dump(metric_dict,open("metrics.json","w"))

def main():
    clf = load_model("model.pkl")
    test_df = load_data(r"data/processed/test_tfidf.csv")
    X_test,y_test = split_X_test_y_test(test_df)
    y_pred,y_pred_proba = prediction(X_test,y_test,clf)
    metric_dict = evaluate_model(y_pred,y_test,y_pred_proba)
    save_metrics(metric_dict)


if __name__ == "__main__":
    main()

