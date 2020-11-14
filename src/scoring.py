import os
import json
import numpy as np
import preprocessing as preprocessing
from tensorflow import keras

def init():
    global model
    global vocab
    global max_len
    model = keras.models.load_model(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model'), compile=False)
    with open(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model', 'vocab.json')) as json_file:
        vocab = json.load(json_file)
    with open(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model', 'params.json')) as json_file:
        params = json.load(json_file)
    max_len = params['max_len']

def run(raw_data):
    tweets = np.array(json.loads(raw_data)["data"])
    processed_tweets = preprocessing.process_tweets(tweets, vocab, max_len)
    result = model.predict(processed_tweets).ravel()
    return result.tolist()