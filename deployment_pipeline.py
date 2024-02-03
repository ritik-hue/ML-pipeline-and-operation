import numpy as numpy
import pandas as pd
"""from materializer.custom_materializer import cs_materializer"""
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

docker_settings= DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy:float =0.5
@step
def deployment_trigger(
    accuracy:float,
    config: DeploymentTriggerConfig,
):
    return accuracy > config.min_accuracy    

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@pipeline(enable_cache=True,settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path=r'C:\Users\ADMIN\Downloads\data\data_new\olist_customers_dataset.csv'
    min_accuracy: float=0.5,
    workers: int=1,
    timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df=ingest_df()
    X_train,X_test,y_train,y_test=clean_df(df)
    model=train_model(X_train,X_test,y_train,y_test)
    r2_score,rmse=evaluate_model(model,X_test,y_test)
    deployment_decision=deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
