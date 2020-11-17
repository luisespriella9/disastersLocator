import os
import argparse
import json
import preprocessing as preprocessing
from azureml.core import Dataset, Run, Workspace

def main(source_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    run = None
    try:
        run = Run.get_context()
        workspace = run.experiment.workspace
        dataset = Dataset.get_by_id(workspace=workspace, id=source_path)

    except:
        # running in offline mode
        workspace = Workspace.from_config()
        dataset = Dataset.get_by_name(workspace=workspace, name=source_path)

    tweets_pd = dataset.to_pandas_dataframe()

    # Process Training Data
    tweets_pd.dropna(inplace=True)
    tweets_pd.reset_index()

    tweets_pd, vocab, max_len = preprocessing.process_dataset(tweets_pd, tweets_column='text', target_column='target')

    # save vocab
    language_folder = target_path+'/language/'
    if not os.path.exists(language_folder):
        os.makedirs(language_folder)
    with open(os.path.join(language_folder, 'vocab.json'), 'w') as f:
        f.write(json.dumps(vocab))

    # save language params
    language_params = {'max_len': max_len}
    with open(os.path.join(language_folder, 'params.json'), 'w') as f:
        f.write(json.dumps(language_params))

    # save train, text adn validation datasets
    processed_data_folder = target_path+'/processed_tweets/'
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)
    tweets_pd.to_csv(path_or_buf=processed_data_folder+"tweets.csv", index=False)

if __name__ == "__main__":
    # Default Path when running offline
    default_target_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

    parser = argparse.ArgumentParser(description='data cleaning for tweets')
    parser.add_argument('-s', '--source_path', help='directory to dataset', default='disaster_tweets_train')
    parser.add_argument('-t', '--target_path', help='directory to cleaned data', default=default_target_path)
    args = parser.parse_args()

    params = vars(args)

    main(**params)