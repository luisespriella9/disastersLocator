import os
import argparse
import json
import pandas as pd
import preprocessing as preprocessing
import tensorflow as tf
from azureml.core import Run
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def split_input_and_targets(dataset):
    return dataset['text'].values, dataset['target'].values

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

def main(source_path, target_path, epochs, batch_size):
    run = Run.get_context()

    # load vocab
    language_folder = source_path+'/language/'
    with open(language_folder+'vocab.json', 'r') as json_file:
        vocab = json.load(json_file)

    # load language params
    with open(language_folder+'params.json', 'r') as json_file:
        language_params = json.load(json_file)
        max_len = language_params['max_len']

    # load training, test and validation data
    processed_data_folder = source_path+'/processed_tweets/'
    tweets_pd = pd.read_csv(processed_data_folder+'tweets.csv')

    train, test = train_test_split(tweets_pd, test_size=0.4, random_state=42, shuffle=True)
    test, validation = train_test_split(test, test_size=0.5, random_state=42, shuffle=True)

    # split text and targets
    X_train, y_train = split_input_and_targets(train)
    X_test, y_test = split_input_and_targets(test)
    X_validation, y_validation = split_input_and_targets(validation)

    # convert text to sequences
    X_train = preprocessing.convert_to_sequence(X_train, vocab, max_len)
    X_test = preprocessing.convert_to_sequence(X_test, vocab, max_len)
    X_validation = preprocessing.convert_to_sequence(X_validation, vocab, max_len)

    model = TweetsDisasterClassifier(vocab_size=len(vocab), embedding_dim=100, max_length=max_len)
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=batch_size, epochs=epochs)

    # Measure Test Results
    y_pred = model.predict(X_test)
    y_pred = [0 if pred < 0.5 else 1 for pred in y_pred]

    accuracy = accuracy_score(y_test, y_pred)

    # Log Metrics
    run.log('accuracy', accuracy)

    # save model
    os.makedirs(target_path, exist_ok=True)
    model_json = model.to_json()
    with open(os.path.join(target_path, 'model.json'), 'w') as f:
        f.write(model_json)

    model.save(target_path)

if __name__ == "__main__":
    # Default Paths when running offline
    outputs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    models_path = os.path.join(outputs_path, 'model')

    parser = argparse.ArgumentParser(description='data cleaning for tweets')
    parser.add_argument('-s', '--source_path', help='directory to training data', default=outputs_path)
    parser.add_argument('-t', '--target_path', help='directory to previous data step', default=models_path)
    parser.add_argument('-e', '--epochs', help='number of epochs', default=2, type=int)
    parser.add_argument('-b', '--batch_size', help='batch size', default=16, type=int)
    args = parser.parse_args()

    params = vars(args)

    main(**params)