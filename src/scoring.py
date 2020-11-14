import os
import joblib
import json
import numpy as np
from azureml.core import Workspace
from azureml.core.model import Model
from tensorflow import keras
from keras.models import model_from_json
from azure.storage.blob import BlobClient
import preprocessing as preprocessing

def init():
    global model
    global vocab
    global max_len
    '''
    json_file = open(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model/model.json'), 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    '''
    model = keras.models.load_model(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model'), compile=False)
    with open(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model', 'vocab.json')) as json_file:
        vocab = json.load(json_file)
    with open(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model', 'params.json')) as json_file:
        params = json.load(json_file)
    max_len = params['max_len']
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def run(raw_data):
    tweets = np.array(json.loads(raw_data)["data"])
    processed_tweets = preprocessing.process_all_tweets(tweets, vocab, max_len)
    result = model.predict(processed_tweets).ravel()
    # You can return any data type, as long as it is JSON serializable.
    return result.tolist()