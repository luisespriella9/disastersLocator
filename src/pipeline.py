import os
import json
import argparse
from azureml.core import Dataset, Datastore, RunConfiguration, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.environment import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline, PipelineEndpoint
from azureml.pipeline.steps import PythonScriptStep

##############################################################
#    Get Azure Machine Learning Resources                    #
##############################################################

def get_workspace(subscription_id: str, resource_group: str, workspace_name: str, tenant_id: str, client_id: str, client_secret: str) -> Workspace:
    if (tenant_id and client_id and client_secret):
        auth = ServicePrincipalAuthentication(tenant_id=tenant_id,
                                                  service_principal_id=client_id,
                                                  service_principal_password=client_secret)
    else:
        auth=None
    return Workspace(subscription_id, resource_group, workspace_name, auth=auth)

def get_datastore(ws: Workspace, datastore_name: str):
    return ws.datastores[datastore_name]

def get_compute(ws: Workspace, compute_name: str):
    compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
    compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")
    if compute_name not in ws.compute_targets:
        provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size, min_nodes=compute_min_nodes, max_nodes=compute_max_nodes)
        compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    return ws.compute_targets[compute_name]

##############################################################
#    Create Pipeline Steps                                   #
##############################################################

def process_step(ws: Workspace, datastore: Datastore, compute: ComputeTarget, run_config: RunConfiguration) -> (PipelineData, PythonScriptStep):
    input_data = Dataset.get_by_name(ws, name='disaster_tweets_train')
    source_path = input_data.as_named_input('disaster_tweets_prepare')
    target_path = PipelineData("prepared_disaster_tweets", datastore=datastore).as_dataset()

    prep_step = PythonScriptStep(script_name="prep.py",
                                inputs=[source_path],
                                outputs=[target_path],
                                arguments=['--source_path', source_path,
                                    '--target_path', target_path],
                                compute_target=compute,
                                source_directory='src',
                                runconfig=run_config)
    return target_path, prep_step

def train_step(datastore: Datastore, input_data: PipelineData, compute: ComputeTarget, run_config: RunConfiguration) -> (PipelineData, PythonScriptStep):
    output_data = PipelineData("train", datastore=datastore).as_dataset()
    
    args = [
        '--source_path', input_data,
        '--target_path', output_data,
        '--batch_size', 16,
       '--epochs', 2]
    train_step = PythonScriptStep(script_name="train_cnn.py",
                                inputs=[input_data.as_named_input('disaster_tweets_train')],
                                arguments=args,
                                outputs=[output_data],
                                compute_target=compute,
                                source_directory='src',
                                runconfig=run_config)
    return output_data, train_step

def register_step(datastore: Datastore, input_data: PipelineData, compute: ComputeTarget, run_config: RunConfiguration) -> (PipelineData, PythonScriptStep):
    output_data = PipelineData("model", datastore=datastore).as_dataset()

    register_step = PythonScriptStep(script_name="register.py",
                                inputs=[input_data.as_named_input('disaster_tweets_register')],
                                outputs=[output_data],
                                arguments=['--source_path', input_data],
                                compute_target=compute,
                                source_directory='src',
                                runconfig=run_config)
    return output_data, register_step

##############################################################
#    Manage Endpoint                                         #
##############################################################

def add_endpoint(ws: Workspace, pipeline: PublishedPipeline, endpoint_name: str) -> PipelineEndpoint:
    endpoint_list = [p.name for p in PipelineEndpoint.list(ws)]
    if endpoint_name in endpoint_list:
        endpoint = PipelineEndpoint.get(workspace=ws, name=endpoint_name)
        endpoint.add_default(pipeline)
    else:
        endpoint = PipelineEndpoint.publish(workspace=ws, name=endpoint_name,
                                                pipeline=pipeline, description="Seer Pipeline Endpoint")
    return endpoint

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

##############################################################
#    Main Run                                                #
##############################################################

if __name__ == "__main__":
    # get secrets
    parser = argparse.ArgumentParser(description='Seer Pipeline')
    parser.add_argument('-a', '--arguments', help='json file with arguments')
    args = parser.parse_args()
    secrets = parse_args(args.arguments)
    
    # get AML Workspace
    ws = get_workspace(secrets["subscription_id"], secrets["resource_group"], secrets["workspace_name"], secrets["tenantId"], secrets["clientId"], secrets["clientSecret"])

    # get datastore
    datastore = get_datastore(ws, 'disaster_tweets')

    # get compute
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "nlp-cpu-cluster")
    compute = get_compute(ws, compute_name)

    # create Experiment
    experiment_name = 'cnn-1convolutional-layer'

    # preparation step
    conda_dependencies = CondaDependencies(conda_dependencies_file_path='conda_dependencies.yml')
    run_config = RunConfiguration(conda_dependencies=conda_dependencies)

    preparation_data, preparation_step = process_step(ws, datastore, compute, run_config)

    # train step
    training_data, train_step = train_step(datastore, preparation_data, compute, run_config)

    # registration step
    _, registration_step = register_step(datastore, training_data, compute, run_config)

    # create pipeline from steps
    pipeline = Pipeline(workspace=ws, steps=[preparation_step, train_step, registration_step])
    published_pipeline = pipeline.publish(name="NLP Disaster Tweets Pipeline", 
        description="Predict Disaster from Tweets")

    # add pipeline to endpoint
    endpoint = add_endpoint(ws, published_pipeline, 'distweets-endpoint')

    # run pipeline
    endpoint.submit(experiment_name)    