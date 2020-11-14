import os
import sys
import matplotlib.pyplot as plt
import preprocessing as preprocessing
import joblib
import json
import tensorflow as tf
from azureml.core import Dataset, Run, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from azure.storage.blob import BlobClient

run = Run.get_context()
workspace = run.experiment.workspace
keyvault = workspace.get_default_keyvault()
storage_account_connection_string = keyvault.get_secret("storage-account-connection-string")

# Get a dataset by name
dataset = Dataset.get_by_name(workspace=workspace, name='disaster_tweets_train')

# Load a TabularDataset into pandas DataFrame
tweets_pd = dataset.to_pandas_dataframe()

# Process Training Data
tweets_pd.dropna(inplace=True)
tweets_pd.reset_index()

train, test = train_test_split(tweets_pd, test_size=0.4, random_state=42, shuffle=True)
test, validation = train_test_split(test, test_size=0.5, random_state=42, shuffle=True)

X_train, y_train, vocab, max_len = preprocessing.process_dataset(train, tweets_column='text', target_column='target')
X_test, y_test, _, _ = preprocessing.process_dataset(test, vocab, max_len, tweets_column='text', target_column='target')
X_validation, y_validation, _, _ = preprocessing.process_dataset(validation, vocab, max_len, tweets_column='text', target_column='target')

# CNN Model
def TweetsDisasterClassifier(vocab_size = 1000, embedding_dim=100, max_length=100):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=10, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = TweetsDisasterClassifier(vocab_size=len(vocab), embedding_dim=100, max_length=max_len)
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=16, epochs=2)

# Measure Test Results
y_pred = model.predict(X_test)
y_pred = [0 if pred < 0.5 else 1 for pred in y_pred]

accuracy = accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

# Log Metrics
run.log('accuracy', accuracy)

# save model
os.makedirs('./outputs/model', exist_ok=True)
model_json = model.to_json()
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)

#model.save_weights('./outputs/model/cnn.h5')
model.save('./outputs/model')

# save vocab
with open('./outputs/model/vocab.json', 'w') as f:
    f.write(json.dumps(vocab))

# save language params
language_params = {'max_len': max_len}
with open('./outputs/model/params.json', 'w') as f:
    f.write(json.dumps(language_params))
'''
# save vocab
vocab_blob = BlobClient.from_connection_string(conn_str=storage_account_connection_string, container_name="language", blob_name="vocab.json")
vocab_blob.upload_blob(json.dumps(vocab), overwrite=True)

# save language model parameters
language_params = {'max_len': max_len}
params_blob = BlobClient.from_connection_string(conn_str=storage_account_connection_string, container_name="language", blob_name="params.json")
params_blob.upload_blob(json.dumps(language_params), overwrite=True)
'''