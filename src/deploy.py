import os
from azureml.core import Environment, Run, Workspace
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice

def main():
    try:
        run = Run.get_context()
        workspace = run.experiment.workspace
    except:
        # running in offline mode
        workspace = Workspace.from_config()

    # get env
    keras_env = Environment.from_conda_specification(name='keras-env', file_path='../conda_dependencies.yml')

    # config
    script_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_config = InferenceConfig(source_directory=script_folder, entry_script='scoring.py', environment=keras_env)
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "Tweets",  "method" : "keras"}, 
                                               description='Predict Disaster from Tweets with keras')

    model = Model(workspace, 'disaster_predictor_cnn')
    service = Model.deploy(
            workspace = workspace,
            name = "disaster-predictor-cnn",
            models = [model],
            inference_config = inference_config,
            deployment_config = deployment_config,
            overwrite=True)

    service.wait_for_deployment(show_output=False)

if __name__ == "__main__":
    main()