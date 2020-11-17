import os
import argparse
from azureml.core import Run

def main(source_path):
    try:
        run = Run.get_context()
        model_name = "disaster_predictor_cnn"
        run.register_model(model_name=model_name, model_path=source_path)
    except:
        print("cannot register model in offline mode")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='registering model')
    parser.add_argument('-s', '--source_path', help='directory to model', default=None)
    args = parser.parse_args()

    params = vars(args)