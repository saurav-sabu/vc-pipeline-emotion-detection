import pandas as pd
import numpy as np
import pickle
import yaml

from sklearn.ensemble import GradientBoostingClassifier

def load_params(params_path):
    params = yaml.safe_load(open(params_path,"r"))["model_building"]
    return params

def load_data(train_bow_path):
    train_data = pd.read_csv(train_bow_path)
    return train_data

def get_X_train_y_train(train_data):
    X_train = train_data.iloc[:,:-1].values
    y_train = train_data.iloc[:,-1].values
    return X_train,y_train

def model_training(X_train,y_train,params):
    gb = GradientBoostingClassifier(n_estimators=params["n_estimators"],learning_rate=params["learning_rate"])
    gb.fit(X_train,y_train)
    return gb

def save_model(gb):
    pickle.dump(gb,open("model.pkl","wb"))

def main():
    params = load_params("params.yaml")
    train_data_bow = load_data(r"data/processed/train_bow.csv")
    X_train, y_train = get_X_train_y_train(train_data_bow)
    gb = model_training(X_train, y_train,params)
    save_model(gb)

if __name__ == "__main__":
    main()
