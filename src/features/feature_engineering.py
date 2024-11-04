import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import yaml

def load_params(param_path):
    max_features = yaml.safe_load(open(param_path,"r"))["feature_engineering"]["max_features"]
    return max_features

def load_data(train_path,test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def remove_na(train_data,test_data):
    train_data.fillna("",inplace=True)
    test_data.fillna("",inplace=True)
    return train_data,test_data

def split_train_test(train_data,test_data):
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
    return X_train,y_train,X_test,y_test

def apply_feature_engineering(X_train,y_train,X_test,y_test,max_features):
    # Apply Tfidf (TfidfVectorizer)
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test

    return train_df,test_df

def save_data(data_path:str,train_df,test_df):
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path,"train_tfidf.csv"),index=False)
        test_df.to_csv(os.path.join(data_path,"test_tfidf.csv"),index=False)
    except Exception as e:
        print("Error Occurred while saving the data")
        print(e)
        raise

def main():
    max_features = load_params("params.yaml")
    train_data, test_data = load_data(r"data/interim/train_processed.csv",r"data/interim/test_processed.csv")
    train_data, test_data = remove_na(train_data,test_data)
    X_train,y_train,X_test,y_test = split_train_test(train_data,test_data)
    train_df, test_df = apply_feature_engineering(X_train,y_train,X_test,y_test,max_features)
    data_path = os.path.join("data","processed")
    save_data(data_path,train_df,test_df)


if __name__ == "__main__":
    main()



