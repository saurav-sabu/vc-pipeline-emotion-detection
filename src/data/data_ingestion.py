import pandas as pd
import numpy as np

import yaml
import os
import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def load_params(params_path:str) -> float:
    try:
        test_size = yaml.safe_load(open(params_path,"r"))["data_ingestion"]["test_size"]
        return test_size
    except Exception as e:
        logger.error(f"Error loading test size from {params_path}: {e}")
        print("Error Occurred while loading the parameters")
        print(e)
        raise
        

def read_data(url:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print("Error Occurred while loading the data")
        print(e)
        raise

def initial_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["tweet_id"],inplace=True)
        final_df = df[df["sentiment"].isin(["neutral","sadness"])]
        final_df["sentiment"].replace({"neutral":1,"sadness":0},inplace=True)
        return final_df
    except Exception as e:
        print("Error Occurred while preprocessing the data")
        print(e)
        raise


def save_data(data_path:str,train_data: pd.DataFrame,test_data: pd.DataFrame)->None:
    try:
        os.makedirs(data_path,exist_ok=True)
        train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)
    except Exception as e:
        print("Error Occurred while saving the data")
        print(e)
        raise

def main():
    test_size = load_params("params.yaml")
    df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/refs/heads/main/tweet_emotions.csv")
    final_df = initial_preprocess(df)
    train_data, test_data = train_test_split(final_df,test_size=test_size,random_state=42)
    data_path = os.path.join("data","raw")
    save_data(data_path,train_data,test_data)


if __name__ == "__main__":
    main()





