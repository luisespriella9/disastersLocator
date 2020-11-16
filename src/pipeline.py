import os
import json
import argparse
from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.widgets import RunDetails
from pathlib import Path

def get_workspace(subscription_id: str, resource_group: str, workspace_name: str, tenant_id: str, client_id: str, client_secret: str) -> Workspace:
    svcprincipal = ServicePrincipalAuthentication(tenant_id=tenant_id,
                                                  service_principal_id=client_id,
                                                  service_principal_password=client_secret)
    return Workspace(subscription_id, resource_group, workspace_name, auth=svcprincipal)

def get_compute(ws: Workspace, compute_name: str):
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")
    if compute_name not in ws.compute_targets:
        provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size, min_nodes=compute_min_nodes, max_nodes=compute_max_nodes)
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    return ws.compute_targets[compute_name]

def create_experiment(ws: Workspace, experiment_name: str):
    experiment = Experiment(workspace=ws, name=experiment_name)
    return experiment

def get_environment(env_name: str):
    env = Environment.from_conda_specification(name=env_name, file_path='conda_dependencies.yml')
    return env

def train_step(compute_target: ComputeTarget, env: Environment, experiment: Experiment):
    args = [
        '--batch-size', 16,
       '--epochs', 2]

    src = ScriptRunConfig(source_directory=os.path.join(Path(os.getcwd()).resolve(), 'src'),
                        script='train_cnn.py',
                        arguments=args,
                        compute_target=compute_target,
                        environment=env)

    src.run_config.source_directory_data_store = "workspaceblobstore" 
    run = experiment.submit(src)
    run.wait_for_completion(show_output=True)
    return run

def register_step(model_name: str, run: RunDetails):
    run.register_model(model_name=model_name, model_path = "outputs/model")


def parse_args(secrets: str) -> dict:
    args = {
        "subscription_id": "",
        "resource_group": "",
        "workspace_name": "",
        "tenantId": "",
        "clientId": "",
        "clientSecret": ""
    }
    print(secrets)
    variables = json.loads(secrets)
    for k,v in variables.items():
        if k in args:
            args[k] = v
    return args

if __name__ == "__main__":
    # get secrets
    parser = argparse.ArgumentParser(description='Seer Pipeline')
    parser.add_argument('-a', '--arguments', help='json file with arguments')
    args = parser.parse_args()
    secrets = parse_args(args.arguments)
    
    # get AML Workspace
    ws = get_workspace(secrets["subscription_id"], secrets["resource_group"], secrets["workspace_name"])

    # get compute
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "nlp-cpu-cluster")
    compute = get_compute(ws, compute_name)

    # create Experiment
    experiment_name = 'cnn-1convolutional-layer'
    experiment = create_experiment(ws, experiment_name)

    # get Environment
    env = get_environment('keras-env')

    # train
    run = train_step(compute, env, experiment)

    # register model
    model_name = "disaster_predictor_cnn"
    register_step(model_name, run)