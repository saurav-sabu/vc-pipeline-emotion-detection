import numpy as np
import pandas as pd
import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


def load_data(train_data_path,test_data_path):

    nltk.download('wordnet')
    nltk.download('stopwords')

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    return train_data, test_data


def lemmatization(text):
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def data_preprocess(train_data, test_data):
    train_preprocessed_data = normalize_text(train_data)
    test_preprocessed_data = normalize_text(test_data)

    return train_preprocessed_data, test_preprocessed_data

def save_data(data_path:str,train_preprocessed_data: pd.DataFrame,test_preprocessed_data: pd.DataFrame)->None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_preprocessed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
    except Exception as e:
        print("Error Occurred while saving the data")
        print(e)
        raise

def main():
    train_data, test_data = load_data(r"data/raw/train.csv",r"data/raw/test.csv")
    train_preprocessed_data, test_preprocessed_data = data_preprocess(train_data,test_data)
    data_path = os.path.join("data","interim")
    save_data(data_path,train_preprocessed_data,test_preprocessed_data)

if __name__ == "__main__":
    main()



