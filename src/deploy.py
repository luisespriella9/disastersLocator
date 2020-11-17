import os
import argparse
import json
from azureml.core import Environment, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice

def get_workspace(subscription_id: str, resource_group: str, workspace_name: str, tenant_id: str, client_id: str, client_secret: str) -> Workspace:
    if (tenant_id and client_id and client_secret):
        auth = ServicePrincipalAuthentication(tenant_id=tenant_id,
                                                  service_principal_id=client_id,
                                                  service_principal_password=client_secret)
    else:
        auth=None
    return Workspace(subscription_id, resource_group, workspace_name, auth=auth)

def parse_args(secrets: str) -> dict:
    args = {
        "subscription_id": "",
        "resource_group": "",
        "workspace_name": "",
        "tenantId": "",
        "clientId": "",
        "clientSecret": ""
    }
    
    if os.path.exists("config.json"):
        with open("config.json") as f:
            variables = json.load(f)
    else:
        variables = json.loads(secrets)
    for k,v in variables.items():
        if k in args:
            args[k] = v
    return args

def main(workspace: Workspace):
    # get env
    keras_env = Environment.from_conda_specification(name='keras-env', file_path='conda_dependencies.yml')

    # config
    inference_config = InferenceConfig(source_directory='.', entry_script='scoring.py', environment=keras_env)
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
    # get secrets
    parser = argparse.ArgumentParser(description='Seer Pipeline')
    parser.add_argument('-a', '--arguments', help='json file with arguments')
    args = parser.parse_args()
    secrets = parse_args(args.arguments)
    
    # get AML Workspace
    ws = get_workspace(secrets["subscription_id"], secrets["resource_group"], secrets["workspace_name"], secrets["tenantId"], secrets["clientId"], secrets["clientSecret"])

    main(ws)